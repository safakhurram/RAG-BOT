"""File loading: turn a PDF/TXT/CSV path into LangChain Documents.

Scanned PDF pages (no text layer) are OCR'd. What matters for getting answers
that match the PDF exactly:

1. OCR engine: the `rapidocr` package (v3) ships multilingual PP-OCRv6 models
   that keep the spaces between English words. The older
   `rapidocr_onnxruntime` package uses a Chinese model that glues words
   together, which breaks both retrieval and answers. It is only a fallback.
2. Resolution: pages are rendered at 150 DPI (RAG_OCR_DPI; match your scan's DPI,
   rendering higher only upscales the same pixels and is slower) and the engine is
   told NOT to shrink them (Global.max_side_len). RapidOCR's default silently
   downscales any image whose long side is over 2000 px, which blurs small
   print (fee tables, footnotes) on a 200 DPI A4 page and causes misread digits.
3. Scan clean-up: pages are converted to grayscale and washed-out scans get
   their contrast stretched before OCR (RAG_OCR_PREPROCESS=0 turns it off).
4. A page is OCR'd when it has no text layer, when its text layer is junk
   (garbled hidden text left by a bad scanner OCR), or ALWAYS when
   RAG_FORCE_OCR=1 (use this when the whole PDF is scanned pictures).
5. Reading order: OCR boxes are re-ordered column by column (ocr_layout.py).
6. Repeated page furniture (headers/footers) is removed.
7. Pages WITH a text layer can still hold text that exists only as a picture;
   large embedded images are OCR'd and their new lines appended.
8. Sideways scans are detected and re-read rotated by 90/270 degrees.
9. Every OCR'd page gets `ocr_conf`, so low-quality pages can be flagged.
10. One page failing in OCR no longer aborts the whole upload.

Raw OCR boxes are cached per file AND per OCR settings (the cache file name
contains a hash of the settings), so changing DPI/threshold re-reads the PDF,
while improving the layout step does not.
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
from langchain_community.document_loaders import CSVLoader, TextLoader
from langchain_core.documents import Document

from .ocr_layout import order_boxes, to_boxes

BASE_DIR = Path(__file__).resolve().parent.parent  # the 01-rag-chatbot folder, not the CWD


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


# ---- OCR settings (all can be overridden from .env) -------------------------
CACHE_DIR = Path(os.getenv("RAG_OCR_CACHE_DIR", BASE_DIR / "ocr_cache"))
OCR_DPI = int(_env_float("RAG_OCR_DPI", 150))  # match the scan: UGProspectus2025 is 150 DPI
OCR_MAX_SIDE = int(_env_float("RAG_OCR_MAX_SIDE", 4000))  # engine default 2000 downscales
OCR_TEXT_SCORE = _env_float("RAG_OCR_TEXT_SCORE", 0.4)  # engine default 0.5 drops faint text
OCR_BOX_THRESH = _env_float("RAG_OCR_BOX_THRESH", 0.4)  # engine default 0.5
PREPROCESS = os.getenv("RAG_OCR_PREPROCESS", "1") != "0"
FORCE_OCR = os.getenv("RAG_FORCE_OCR", "auto").strip().lower() in {"1", "true", "yes", "all"}

_SETTINGS_ID = hashlib.sha1(
    f"{OCR_DPI}|{OCR_MAX_SIDE}|{OCR_TEXT_SCORE}|{OCR_BOX_THRESH}|{PREPROCESS}".encode()
).hexdigest()[:8]
CACHE_VERSION = f"boxes-v3-{_SETTINGS_ID}"  # older caches (200 DPI, downscaled) are ignored

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".csv"}
MIN_TEXT_CHARS = 20  # a page with fewer characters is treated as "no text layer"
REPEATED_LINE_SHARE = 0.4  # a line on >= 40% of pages is a header/footer
MIN_PAGES_FOR_FURNITURE = 8  # in a short file a repeated line is content, not a header

# Deployed servers: refuse uploads that would need more live OCR than this many
# pages (0 = no limit). Pages already in the OCR cache do not count.
MAX_ONLINE_OCR_PAGES = int(_env_float("RAG_MAX_ONLINE_OCR_PAGES", 0))

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

            tuned = {
                "Global.log_level": "error",
                "Global.max_side_len": OCR_MAX_SIDE,
                "Global.text_score": OCR_TEXT_SCORE,
                "Det.box_thresh": OCR_BOX_THRESH,
            }
            try:
                _ocr_engine = RapidOCR(params=tuned)
            except Exception as exc:  # a key this rapidocr version doesn't know
                print(
                    f"WARNING: could not apply tuned OCR settings ({exc}); using defaults.",
                    flush=True,
                )
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


def _preprocess(img: np.ndarray) -> np.ndarray:
    """Grayscale + contrast stretch for faded scans. Never changes the geometry."""
    if not PREPROCESS:
        return img
    try:
        import cv2
    except ImportError:
        return img
    if img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img.ndim == 3:
        gray = img[:, :, 0]
    else:
        gray = img
    lo, hi = np.percentile(gray[::4, ::4], (1, 99))
    if hi - lo >= 1 and hi - lo < 200:  # washed-out / grey scan
        stretched = (gray.astype(np.float32) - lo) * (255.0 / (hi - lo))
        gray = np.clip(stretched, 0, 255).astype(np.uint8)
    return cv2.cvtColor(np.ascontiguousarray(gray), cv2.COLOR_GRAY2RGB)


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
    img = _preprocess(img)
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
    pix = page.get_pixmap(dpi=OCR_DPI, colorspace=fitz.csRGB, alpha=False, clip=clip)
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


class _BoxCache:
    """JSON cache of raw OCR results. Written every few pages (not after every
    page, which re-wrote the whole file hundreds of times) and atomically, so a
    crash cannot leave a half-written file; a corrupt file is simply ignored."""

    def __init__(self, file: Path):
        self.file = file
        self.dirty = 0
        try:
            self.data = json.loads(file.read_text(encoding="utf-8")) if file.exists() else {}
        except (OSError, ValueError):
            self.data = {}

    def get(self, key: str, make) -> dict:
        if key not in self.data:
            self.data[key] = make()
            self.dirty += 1
            if self.dirty >= 5:
                self.flush()
        return self.data[key]

    def flush(self) -> None:
        if not self.dirty:
            return
        self.file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.file.with_name(self.file.name + ".tmp")
        tmp.write_text(json.dumps(self.data), encoding="utf-8")
        tmp.replace(self.file)
        self.dirty = 0


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


def _text_layer_is_junk(text: str) -> bool:
    """True when a page's embedded text is missing or unusable (garbled hidden
    text from a bad scanner OCR), so the page must be OCR'd ourselves."""
    chars = [c for c in text if not c.isspace()]
    if len(chars) < MIN_TEXT_CHARS:
        return True
    bad = sum(1 for c in chars if c == "\ufffd" or ord(c) < 32 or 0xE000 <= ord(c) <= 0xF8FF)
    if bad / len(chars) > 0.05:
        return True
    if sum(c.isalpha() for c in chars) / len(chars) < 0.25:
        return True
    tokens = text.split()
    if len(tokens) >= 20 and sum(len(t) == 1 for t in tokens) / len(tokens) > 0.4:
        return True  # "N E D  U N I V E R S I T Y" style letter-spacing garbage
    return False


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


def _pdf_pages(path: Path) -> list[Document]:
    """One Document per PDF page with the embedded text layer (empty for pure
    pictures). PyMuPDF is used instead of pypdf: it is faster and does not
    choke on scanner-produced files. `page` is 0-based like PyPDFLoader's."""
    docs = []
    with fitz.open(str(path)) as pdf:
        if pdf.needs_pass:
            raise ValueError(f"{path.name} is password-protected.")
        for i, page in enumerate(pdf):
            docs.append(
                Document(
                    page_content=page.get_text("text") or "",
                    metadata={"source": str(path), "page": i},
                )
            )
    return docs


def _load_pdf(path: Path, on_progress=None) -> list[Document]:
    docs = _pdf_pages(path)
    cache = _BoxCache(CACHE_DIR / f"{_file_hash(path)}.{CACHE_VERSION}.json")

    with fitz.open(str(path)) as pdf:
        # plan: (doc, page index, None) = OCR the whole page,
        #       (doc, page index, regions) = OCR pictures inside a text page
        plan = []
        for d in docs:
            idx = int(d.metadata.get("page", 0))
            if FORCE_OCR or _text_layer_is_junk(d.page_content):
                plan.append((d, idx, None))
            elif OCR_EMBEDDED_IMAGES:
                regions = _embedded_image_regions(pdf[idx])
                if regions:
                    plan.append((d, idx, regions))

        total = len(plan)
        if MAX_ONLINE_OCR_PAGES:
            uncached = sum(
                1 for _, idx, regions in plan if regions is None and str(idx) not in cache.data
            )
            if uncached > MAX_ONLINE_OCR_PAGES:
                raise ValueError(
                    f"{path.name} has {uncached} scanned pages that are not in the OCR cache. "
                    f"This server only allows {MAX_ONLINE_OCR_PAGES} pages of live OCR. "
                    "Use the exact PDF the cache was built from, or run the OCR offline first."
                )
        if total:
            n_scans = sum(1 for _, _, r in plan if r is None)
            print(
                f"{path.name}: {len(docs)} pages, {n_scans} scanned, "
                f"{total - n_scans} with embedded images (dpi={OCR_DPI})",
                flush=True,
            )
        start = time.time()
        try:
            for n, (d, idx, regions) in enumerate(plan, 1):
                _report(
                    on_progress,
                    f"Reading {path.name}: page {idx + 1} ({n}/{total})",
                    (n - 1) / total,
                )
                page = pdf[idx]
                try:
                    if regions is None:  # scanned page: OCR everything
                        data = cache.get(str(idx), lambda: _ocr_page_boxes(page))
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
                            data = cache.get(key, lambda: _ocr_image(_page_image(page, clip=rect)))
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
                except Exception as exc:  # one bad page must not kill the whole upload
                    d.metadata["ocr_failed"] = True
                    print(f"WARNING: OCR failed on page {idx + 1} of {path.name}: {exc!r}", flush=True)
                if n % 10 == 0 or n == total:
                    print(f"OCR {n}/{total} ({time.time() - start:.0f}s)", flush=True)
        finally:
            cache.flush()

    _strip_repeated_lines(docs)
    for d in docs:
        # 1-based page number as shown in a PDF viewer
        d.metadata["page_number"] = int(d.metadata.get("page", 0)) + 1

    kept = [d for d in docs if d.page_content.strip()]
    empty = [d.metadata["page_number"] for d in docs if not d.page_content.strip()]
    if empty:
        print(f"{path.name}: no text found on pages {empty[:20]}{'...' if len(empty) > 20 else ''}", flush=True)
    return kept


def low_confidence_pages(docs: list[Document]) -> list[str]:
    """Human-readable warnings for pages whose OCR looks unreliable."""
    by_file: dict[str, list[int]] = {}
    failed: dict[str, list[int]] = {}
    for d in docs:
        name = d.metadata.get("source", "document")
        page = d.metadata.get("page_number")
        conf = d.metadata.get("ocr_conf")
        if conf is not None and conf < LOW_OCR_CONFIDENCE and page:
            by_file.setdefault(name, []).append(page)
        if d.metadata.get("ocr_failed") and page:
            failed.setdefault(name, []).append(page)
    out = []
    for name, pages in by_file.items():
        shown = ", ".join(map(str, pages[:15])) + ("..." if len(pages) > 15 else "")
        out.append(f"{name}: text on page(s) {shown} was hard to read (blurry or low-contrast scan). Check those answers.")
    for name, pages in failed.items():
        shown = ", ".join(map(str, pages[:15]))
        out.append(f"{name}: page(s) {shown} could not be read by OCR.")
    return out


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