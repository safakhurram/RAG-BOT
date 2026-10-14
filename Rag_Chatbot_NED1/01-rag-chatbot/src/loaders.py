"""File loading: turn a PDF/TXT/CSV path into LangChain Documents.

Scanned PDF pages (no text layer) are OCR'd. What matters for getting answers
that match the PDF exactly:

1. OCR engine: the `rapidocr` package (v3) ships multilingual PP-OCRv6 models
   that keep the spaces between English words. The older
   `rapidocr_onnxruntime` package uses a Chinese model that glues words
   together ("DeanFacultyofCivilandPetroleumEngineering"), which breaks both
   retrieval and the bot's answers. The old package is only used as a fallback.
2. Reading order: OCR boxes are re-ordered column by column (see
   ocr_layout.py) so two-column pages are not interleaved.
3. Repeated page furniture ("NED UNIVERSITY", "PROSPECTUS 2025",
   "www.neduet.edu.pk", ...) is removed so it doesn't pollute every chunk.
4. Pages WITH a text layer can still contain text that exists only as a
   picture (a fee table pasted as an image, a scanned form). Embedded images
   of a useful size are OCR'd too and their lines are appended to the page,
   skipping lines the text layer already has (no duplicated content).
5. Sideways scans are detected (most text boxes are taller than wide) and
   re-read rotated by 90/270 degrees; the better reading is kept.
6. Every OCR'd page gets `ocr_conf` (mean recogniser confidence, weighted by
   text length) so low-quality pages can be flagged instead of trusted.

Raw OCR boxes are cached per file, so the layout step can be improved and
re-run without OCR'ing the whole PDF again. Cache entries from the previous
version stay valid (new fields are optional).
"""

import hashlib
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document

from .ocr_layout import order_boxes, to_boxes

CACHE_DIR = Path("ocr_cache")
CACHE_VERSION = "boxes-v2"  # older text-only caches are ignored (wrong order, glued words)
OCR_DPI = 200

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".csv"}
MIN_TEXT_CHARS = 20  # a page with fewer characters is treated as "no text layer"
REPEATED_LINE_SHARE = 0.4  # a line on >= 40% of pages is a header/footer
MIN_PAGES_FOR_FURNITURE = 8  # in a short file a repeated line is content, not a header

LOW_OCR_CONFIDENCE = 0.80  # pages below this are flagged to the user

# embedded images worth OCR'ing on pages that already have a text layer
OCR_EMBEDDED_IMAGES = os.getenv("RAG_OCR_EMBEDDED_IMAGES", "1") != "0"
MIN_IMAGE_PX = 120  # smaller pictures are icons/bullets
MIN_IMAGE_AREA_SHARE = 0.02  # share of the page area; smaller = logo
MAX_IMAGE_AREA_SHARE = 0.85  # bigger = a scan behind an existing text layer

_ocr_engine = None
_ocr_engine_name = None

def _report(cb, stage: str, fraction: float) -> None:
    if cb:
        cb(stage, fraction)

def _get_ocr_engine():
    global _ocr_engine, _ocr_engine_name
    if _ocr_engine is None:
        try:
            from rapidocr import RapidOCR  # v3: bundled PP-OCRv6 models, keeps spaces

            _ocr_engine = RapidOCR(params={"Global.log_level": "error"})
            _ocr_engine_name = "rapidocr-v3"
        except ImportError:
            from rapidocr_onnxruntime import RapidOCR

            print(
                "WARNING: using rapidocr_onnxruntime; English words may be glued "
                "together. Run: pip install rapidocr opencv-python-headless",
                flush=True,
            )
            _ocr_engine = RapidOCR()
            _ocr_engine_name = "rapidocr-onnxruntime"
    return _ocr_engine

def _run_ocr(img: np.ndarray):
    """Raw OCR of one image -> (quads, texts, scores)."""
    engine = _get_ocr_engine()
    if _ocr_engine_name == "rapidocr-v3":
        result = engine(img)
        quads = [] if result.boxes is None else result.boxes.tolist()
        texts = list(result.txts or [])
        raw = getattr(result, "scores", None)
        scores = [] if raw is None else [float(s) for s in raw]
    else:
        result, _ = engine(img)
        quads = [r[0] for r in result or []]
        texts = [r[1] for r in result or []]
        scores = [float(r[2]) for r in result or []]
    scores = (scores + [1.0] * len(texts))[: len(texts)]
    return quads, texts, scores

def _looks_sideways(quads) -> bool:
    """Most text boxes much taller than wide => the page is rotated 90 degrees."""
    if len(quads) < 5:
        return False
    tall = 0
    for q in quads:
        xs = [p[0] for p in q]
        ys = [p[1] for p in q]
        if (max(ys) - min(ys)) > 1.5 * (max(xs) - min(xs)):
            tall += 1
    return tall / len(quads) > 0.6

def _quality(texts, scores) -> float:
    return sum(len(t) * s for t, s in zip(texts, scores))

def _ocr_image(img: np.ndarray) -> dict:
    """OCR one image (trying 90/270 degree rotations if it looks sideways).
    Returns {"w", "h", "items": [[quad, text], ...], "scores": [...]}."""
    quads, texts, scores = _run_ocr(img)
    best = (img, quads, texts, scores)
    if _looks_sideways(quads):
        for k in (1, 3):
            rotated = np.ascontiguousarray(np.rot90(img, k))
            q2, t2, s2 = _run_ocr(rotated)
            if _quality(t2, s2) > _quality(best[2], best[3]):
                best = (rotated, q2, t2, s2)
    img, quads, texts, scores = best
    return {
        "w": int(img.shape[1]),
        "h": int(img.shape[0]),
        "items": [[q, t] for q, t in zip(quads, texts)],
        "scores": scores,
    }

def _page_image(page: "fitz.Page", clip=None) -> np.ndarray:
    pix = page.get_pixmap(dpi=OCR_DPI, alpha=False, clip=clip)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

def _ocr_page_boxes(page: "fitz.Page") -> dict:
    """OCR one whole page."""
    return _ocr_image(_page_image(page))

def _mean_conf(data: dict):
    scores = data.get("scores") or []
    texts = [t for _, t in data.get("items", [])]
    if not scores or len(scores) != len(texts):
        return None  # cache entry from an older version
    total = sum(len(t) for t in texts) or 1
    return sum(s * len(t) for s, t in zip(scores, texts)) / total

def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]

def _norm_line(line: str) -> str:
    return re.sub(r"[^a-z]", "", line.lower())

def _dedupe_key(line: str) -> str:
    """Like _norm_line but keeps digits, so "Rs. 855,000" != "Rs. 56,390"."""
    return re.sub(r"[^a-z0-9]", "", line.lower())

def _merge_image_text(page_text: str, image_lines: list[str]) -> str:
    """Append text read from embedded images, skipping lines the page already has."""
    seen = {k for k in (_dedupe_key(l) for l in page_text.splitlines()) if k}
    extra = []
    for line in image_lines:
        key = _dedupe_key(line)
        if key and key not in seen:
            extra.append(line)
            seen.add(key)
    return page_text + "\n" + "\n".join(extra) if extra else page_text

def _embedded_image_regions(page: "fitz.Page") -> list[tuple[int, "fitz.Rect"]]:
    """(xref, on-page rectangle) of pictures big enough to contain readable text."""
    page_area = page.rect.width * page.rect.height
    if page_area <= 0:
        return []
    out = []
    for info in page.get_images(full=True):
        xref, width, height = info[0], info[2], info[3]
        if min(width, height) < MIN_IMAGE_PX:
            continue
        for rect in page.get_image_rects(xref):
            clip = rect & page.rect
            if clip.is_empty:
                continue
            share = clip.width * clip.height / page_area
            if MIN_IMAGE_AREA_SHARE <= share <= MAX_IMAGE_AREA_SHARE:
                out.append((xref, clip))
    return out

def _parts(line: str) -> list[str]:
    """A line and, if it was joined from side-by-side boxes, its " | " parts."""
    parts = [p for p in line.split(" | ") if p.strip()]
    return [line] + (parts if len(parts) > 1 else [])

def _strip_repeated_lines(docs: list[Document]) -> None:
    """Remove headers/footers: lines (ignoring case, spaces, digits) that
    appear on a large share of the pages. A line made of side-by-side parts
    ("NED UNIVERSITY | UNDERGRADUATE") is removed when every part is repeated."""
    if len(docs) < MIN_PAGES_FOR_FURNITURE:
        return
    counts = Counter()
    for d in docs:
        units = {_norm_line(u) for l in d.page_content.splitlines() for u in _parts(l)}
        counts.update(u for u in units if u)
    repeated = {
        k for k, n in counts.items()
        if n >= REPEATED_LINE_SHARE * len(docs) and len(k) >= 6
    }

    def is_furniture(line: str) -> bool:
        norm = _norm_line(line)
        if not norm:
            return False
        if norm in repeated:
            return True
        parts = [_norm_line(p) for p in line.split(" | ") if _norm_line(p)]
        return len(parts) > 1 and all(p in repeated for p in parts)

    for d in docs:
        kept = [l for l in d.page_content.splitlines() if not is_furniture(l)]
        d.page_content = "\n".join(kept).strip()

def _take_printed_page_number(boxes, page_h: float):
    """The number printed in the page footer (e.g. "93") differs from the PDF
    viewer's page number (e.g. 100). Keep it for citations, drop it from text."""
    footer = [
        b for b in boxes
        if re.fullmatch(r"\d{1,3}", b.text.strip()) and b.y0 > 0.88 * page_h
    ]
    if len(footer) != 1:
        return None, boxes
    return int(footer[0].text), [b for b in boxes if b is not footer[0]]

def _load_pdf(path: Path, on_progress=None) -> list[Document]:
    docs = PyPDFLoader(str(path)).load()

    cache_file = CACHE_DIR / f"{_file_hash(path)}.{CACHE_VERSION}.json"
    cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}

    def cached(key: str, make) -> dict:
        if key not in cache:
            cache[key] = make()
            CACHE_DIR.mkdir(exist_ok=True)
            cache_file.write_text(json.dumps(cache), encoding="utf-8")
        return cache[key]

    with fitz.open(str(path)) as pdf:
        # plan: (doc, page index, None) = OCR the whole page,
        #       (doc, page index, regions) = OCR pictures inside a text page
        plan = []
        for d in docs:
            idx = int(d.metadata.get("page", 0))
            if len(d.page_content.strip()) < MIN_TEXT_CHARS:
                plan.append((d, idx, None))
            elif OCR_EMBEDDED_IMAGES:
                regions = _embedded_image_regions(pdf[idx])
                if regions:
                    plan.append((d, idx, regions))

        total = len(plan)
        if total:
            n_scans = sum(1 for _, _, r in plan if r is None)
            print(
                f"{path.name}: {len(docs)} pages, {n_scans} scanned, "
                f"{total - n_scans} with embedded images",
                flush=True,
            )
        start = time.time()
        for n, (d, idx, regions) in enumerate(plan, 1):
            _report(on_progress, f"Reading {path.name}: page {idx + 1} ({n}/{total})", (n - 1) / total)
            page = pdf[idx]

            if regions is None:  # scanned page: OCR everything
                data = cached(str(idx), lambda: _ocr_page_boxes(page))
                boxes = to_boxes(data["items"])
                page_h = data.get("h") or page.rect.height * OCR_DPI / 72
                printed, boxes = _take_printed_page_number(boxes, page_h)
                if printed:
                    d.metadata["printed_page"] = printed
                d.page_content = "\n".join(order_boxes(boxes, data["w"]))
                d.metadata["ocr"] = True
                conf = _mean_conf(data)
                if conf is not None:
                    d.metadata["ocr_conf"] = round(conf, 3)
            else:  # text page with pictures: add only text the page doesn't have
                lines: list[str] = []
                confs: list[float] = []
                for xref, rect in regions:
                    key = f"{idx}:img{xref}:{int(rect.x0)}_{int(rect.y0)}"
                    data = cached(key, lambda: _ocr_image(_page_image(page, clip=rect)))
                    lines.extend(order_boxes(to_boxes(data["items"]), data["w"]))
                    c = _mean_conf(data)
                    if c is not None:
                        confs.append(c)
                merged = _merge_image_text(d.page_content, lines)
                if merged != d.page_content:
                    d.page_content = merged
                    d.metadata["image_ocr"] = True
                    if confs:
                        d.metadata["ocr_conf"] = round(min(confs), 3)
            if n % 10 == 0 or n == total:
                print(f"OCR {n}/{total} ({time.time() - start:.0f}s)", flush=True)

    _strip_repeated_lines(docs)
    for d in docs:
        # 1-based page number as shown in a PDF viewer
        d.metadata["page_number"] = int(d.metadata.get("page", 0)) + 1
    return [d for d in docs if d.page_content.strip()]

def _load_with_encodings(make_loader, path: Path) -> list[Document]:
    """Try UTF-8 (with or without BOM), then Windows-1252, then Latin-1 (never fails)."""
    last = None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return make_loader(str(path), enc).load()
        except UnicodeDecodeError as exc:
            last = exc
    raise last  # pragma: no cover

def load_file(path: str | Path, on_progress=None) -> list[Document]:
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        docs = _load_pdf(path, on_progress)
    elif ext == ".txt":
        docs = _load_with_encodings(lambda p, enc: TextLoader(p, encoding=enc), path)
    elif ext == ".csv":
        docs = _load_with_encodings(lambda p, enc: CSVLoader(p, encoding=enc), path)
    else:
        raise ValueError(
            f"Unsupported file type: {ext!r}. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if not docs:
        raise ValueError(f"No text could be extracted from {path.name}.")

    for doc in docs:
        doc.metadata["source"] = path.name
    return docs

def load_files(paths: list[str | Path], on_progress=None) -> list[Document]:
    docs: list[Document] = []
    total = max(1, len(paths))
    for i, path in enumerate(paths):
        def sub(stage: str, fraction: float, i: int = i) -> None:
            _report(on_progress, stage, (i + fraction) / total)
        docs.extend(load_file(path, on_progress=sub))
    return docs