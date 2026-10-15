"""Streamlit UI: upload files, then ask questions. Answers come ONLY from the uploaded files."""

import ast
import html
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.rag import RAGPipeline

load_dotenv()

APP_NAME = "DocBot"
AVATAR = ":material/auto_awesome:"

st.set_page_config(
    page_title=APP_NAME,
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Optional: ask the model to answer in a Markdown structure that the UI turns
# into numbered section cards. It is OFF by default because the hint is added
# to the question itself, which (a) pollutes the search query used for
# retrieval and (b) forces greetings and polite refusals into a formatted
# document. The renderer below already handles plain answers and lists well.
USE_STRUCTURE_HINT = False
STRUCTURE_HINT = (
    "\n\nFormat the answer in Markdown: start with a '# ' title of at most 6 words, "
    "then one short intro sentence, then 2-4 sections, each beginning with a '## ' heading "
    "followed by short bullet points like '- **Term**: explanation'. "
    "Use a '> ' line only for one key quote from the document. "
    "Use only information from the documents."
)

SUGGESTIONS = [
    "What is the admission fee?",
    "Who is the Vice-Chancellor?",
    "What is the security deposit?",
    "What are the hostel charges?",
]

# ---------------------------------------------------------------------------
# Styling: dark glass UI, warm peach + lavender accents, serif headings
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #0a0c15;
    --glass: rgba(255,255,255,.045);
    --glass-strong: rgba(255,255,255,.075);
    --line: rgba(255,255,255,.11);
    --text: #e9eaf4;
    --muted: #9a9db3;
    --peach: #f4c9a8;
    --lav: #b7a3f2;
    --mint: #6fe3b1;
    --amber: #f2b45a;
    --pdf: #ff6b81;
    --txt: #6ea8ff;
    --csv: #4fd1a5;
}

/* ---------- base ---------- */
.stApp {
    background-color: var(--bg);
    background-image:
        radial-gradient(900px 500px at 78% -5%, rgba(183,163,242,.16), transparent 60%),
        radial-gradient(800px 500px at 10% 100%, rgba(244,201,168,.10), transparent 60%);
    color: var(--text);
}
.stApp, .stApp p, .stApp li, .stApp label, .stApp textarea, .stApp input,
.stApp [data-testid="stMarkdownContainer"] {
    font-family: 'Inter', sans-serif !important;
    color: var(--text);
}
.stApp h1, .stApp h2, .stApp h3 { font-family: 'Cormorant Garamond', serif !important; color: var(--text); }
[data-testid="stCaptionContainer"] { color: var(--muted) !important; }

/* keep Streamlit's icon font intact */
[data-testid="stIconMaterial"], .material-symbols-rounded, span[class*="material"] {
    font-family: 'Material Symbols Rounded' !important;
}

header[data-testid="stHeader"] { background: transparent; }
.stAppDeployButton, [data-testid="stMainMenu"], #MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.2rem; max-width: 1500px; }

/* ---------- top bar ---------- */
.topbar { display: flex; justify-content: flex-end; margin-bottom: 0.8rem; }
.pill {
    display: inline-flex; align-items: center; gap: 8px; padding: 6px 14px;
    border: 1px solid var(--line); border-radius: 999px; background: var(--glass);
    font-size: 0.8rem; color: var(--text);
}
.pill .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--mint); box-shadow: 0 0 8px var(--mint); }
.pill .dot.off { background: var(--amber); box-shadow: 0 0 8px var(--amber); }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0f1b, #0a0c15);
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
section[data-testid="stSidebar"] .stButton > button { width: 100%; }
.brand { display: flex; align-items: center; gap: 12px; margin-bottom: 1.4rem; }
.brand .logo { font-size: 2.2rem; color: var(--peach); line-height: 1; text-shadow: 0 0 18px rgba(244,201,168,.6); }
.brand .name { font-family: 'Cormorant Garamond', serif; font-size: 2rem; font-weight: 600; line-height: 1; }
.brand .tag { font-size: 0.72rem; color: var(--muted); margin-top: 4px; }
.side-title { font-size: 0.85rem; font-weight: 600; margin: 1.1rem 0 0.6rem 0; }

.doc-item { display: flex; align-items: center; gap: 12px; padding: 8px 4px; }
.doc-item .meta { min-width: 0; flex: 1; }
.doc-item .nm { font-size: 0.8rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.doc-item .sz { font-size: 0.7rem; color: var(--muted); }
.doc-item .live { width: 8px; height: 8px; border-radius: 50%; background: var(--mint); box-shadow: 0 0 8px var(--mint); }
.empty-note { font-size: 0.8rem; color: var(--muted); padding: 4px 0; }

.ficon {
    flex: 0 0 auto; width: 34px; height: 34px; border-radius: 9px; display: inline-flex;
    align-items: center; justify-content: center; font-size: 0.6rem; font-weight: 700; letter-spacing: .03em;
    background: rgba(255,255,255,.08); color: #fff; border: 1px solid var(--line);
}
.ficon.pdf { background: rgba(255,107,129,.18); color: var(--pdf); }
.ficon.txt { background: rgba(110,168,255,.18); color: var(--txt); }
.ficon.csv { background: rgba(79,209,165,.18); color: var(--csv); }

.quote {
    margin-top: 1.4rem; padding: 1.2rem; border-radius: 18px; border: 1px solid var(--line); min-height: 150px;
    background:
        radial-gradient(120px 60px at 85% 100%, rgba(183,163,242,.55), transparent 70%),
        radial-gradient(160px 70px at 20% 100%, rgba(244,201,168,.35), transparent 70%),
        var(--glass);
}
.quote .spark { color: var(--peach); font-size: 1.3rem; }
.quote .q { font-family: 'Cormorant Garamond', serif; font-size: 1.35rem; line-height: 1.25; margin-top: 0.4rem; }

/* ---------- buttons ---------- */
.stButton > button {
    background: linear-gradient(135deg, #f3d3bd, #b8a2f0); color: #1a1530; border: none;
    border-radius: 12px; padding: 0.6rem 1rem; font-weight: 600;
    transition: transform .12s, box-shadow .12s;
}
.stButton > button p { color: #1a1530 !important; font-weight: 600; }
.stButton > button:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 6px 22px rgba(183,163,242,.35); color: #1a1530; }
.stButton > button:disabled { background: rgba(255,255,255,.06); }
.stButton > button:disabled p { color: #70748a !important; }

/* suggestion chips */
.st-key-chips .stButton > button {
    background: transparent; border: 1px solid rgba(255,255,255,.28); border-radius: 999px;
    padding: 0.45rem 0.8rem; font-weight: 500;
}
.st-key-chips .stButton > button p { color: var(--text) !important; font-weight: 500; font-size: 0.82rem; }
.st-key-chips .stButton > button:hover:not(:disabled) {
    background: rgba(244,201,168,.10); border: 1px solid var(--peach); box-shadow: none; transform: none;
}

/* ---------- uploader ---------- */
[data-testid="stFileUploaderDropzone"] {
    background: var(--glass); border: 1px dashed rgba(255,255,255,.28); border-radius: 16px;
}
[data-testid="stFileUploaderDropzone"] * { color: var(--muted); }
[data-testid="stFileUploaderDropzone"] button {
    background: transparent; border: 1px solid var(--line); border-radius: 10px; color: var(--text);
}
[data-testid="stFileUploaderDropzone"] button:hover { border-color: var(--peach); color: var(--peach); }
[data-testid="stFileUploaderFile"] { background: var(--glass); border-radius: 10px; }

/* ---------- hero ---------- */
.hero {
    position: relative; overflow: hidden; border-radius: 24px; border: 1px solid var(--line);
    min-height: 270px; margin-bottom: 1rem;
    background:
        radial-gradient(500px 260px at 72% 20%, rgba(244,201,168,.28), transparent 65%),
        radial-gradient(420px 260px at 15% 100%, rgba(183,163,242,.22), transparent 65%),
        linear-gradient(135deg, #1b1830, #0e1020);
}
.hero .arch {
    position: absolute; right: 15%; bottom: 0; width: 150px; height: 230px; border-radius: 90px 90px 0 0;
    background: linear-gradient(180deg, rgba(255,232,210,.92), rgba(244,201,168,.35) 60%, transparent);
    box-shadow: 0 0 70px rgba(244,201,168,.35);
}
.hero .mtn { position: absolute; bottom: 0; background: #14111e; }
.hero .m1 { right: 27%; width: 120px; height: 150px; clip-path: polygon(0 100%, 30% 25%, 55% 60%, 78% 0, 100% 100%); }
.hero .m2 { right: 4%; width: 190px; height: 130px; clip-path: polygon(0 100%, 35% 30%, 60% 70%, 85% 10%, 100% 100%); background: #181425; }
.hero .m3 { right: 0; left: 0; height: 50px; background: linear-gradient(180deg, transparent, rgba(10,12,21,.85)); }
.hero-text { position: relative; z-index: 2; padding: 2.2rem 2.4rem; max-width: 60%; }
.hero-text h1 { font-size: 3.2rem; font-weight: 500; margin: 0 0 0.6rem 0; line-height: 1.05; }
.hero-text h1 .spark { color: var(--peach); font-size: 1.6rem; vertical-align: top; }
.hero-text p { font-size: 1rem; line-height: 1.55; margin: 0; color: var(--text); }
.hero-text .hint { margin-top: 0.9rem; font-size: 0.85rem; color: var(--peach); }

/* ---------- chat bubbles ---------- */
[data-testid="stChatMessage"] {
    background: var(--glass); border: 1px solid var(--line); border-radius: 20px;
    padding: 1rem 1.2rem; margin-bottom: 1rem; max-width: 100%;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse; margin-left: auto; max-width: 70%; background: var(--glass-strong);
}
[data-testid="stChatMessageAvatarAssistant"] {
    background: linear-gradient(135deg, #f3d3bd, #b8a2f0) !important; color: #1a1530 !important; border-radius: 12px;
}
[data-testid="stChatMessageAvatarUser"] { background: rgba(255,255,255,.12) !important; border-radius: 12px; }
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { font-size: 0.95rem; line-height: 1.65; }
.ts { font-size: 0.7rem; color: var(--muted); text-align: right; margin-top: 2px; }

/* ---------- structured answer ---------- */
.ans { width: 100%; }
.ans-head { display: flex; align-items: center; flex-wrap: wrap; gap: 12px; margin-bottom: 0.7rem; }
.ans-title { font-size: 1.25rem; font-weight: 600; letter-spacing: -.01em; }
.gen-badge {
    display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 999px;
    border: 1px solid var(--line); background: var(--glass); font-size: 0.72rem; color: var(--muted);
}
.ans-intro { font-size: 0.95rem; line-height: 1.7; color: #d7d9ea; margin: 0 0 0.4rem 0; max-width: 78ch; }

.ans-sec {
    border: 1px solid var(--line); border-radius: 16px; padding: 1rem 1.2rem;
    background: rgba(255,255,255,.03); margin-top: 0.9rem;
}
.sec-head { display: flex; align-items: center; gap: 12px; margin-bottom: 0.6rem; }
.sec-num {
    flex: 0 0 auto; width: 32px; height: 32px; border-radius: 50%; display: inline-flex; align-items: center;
    justify-content: center; font-weight: 600; font-size: 0.9rem; color: #fff;
    background: rgba(183,163,242,.28); border: 1px solid rgba(183,163,242,.5);
}
.sec-title { font-size: 1.02rem; font-weight: 600; }
.sec-body { padding-left: 44px; }
.sec-body .sp { margin: 0.2rem 0 0.5rem 0; color: #d7d9ea; }
.sec-body strong { color: #fff; font-weight: 600; }
.sec-body em { color: var(--peach); font-style: italic; }
.sec-body code, .ans-intro code {
    font-family: 'JetBrains Mono', monospace; font-size: 0.82em; background: rgba(255,255,255,.08);
    padding: 1px 6px; border-radius: 6px; color: var(--peach);
}

.blist { list-style: none !important; padding: 0 !important; margin: 0.3rem 0 0.4rem 0 !important; }
.blist li { position: relative; padding-left: 26px; margin: 7px 0; color: #d7d9ea; }
.blist.dots li::before {
    content: ""; position: absolute; left: 6px; top: 0.62em; width: 7px; height: 7px; border-radius: 50%;
    background: var(--peach);
}
.blist.checks li::before {
    content: "✓"; position: absolute; left: 0; top: 0.12em; width: 18px; height: 18px; border-radius: 50%;
    border: 1.5px solid var(--mint); color: var(--mint); font-size: 0.65rem; line-height: 15px; text-align: center;
}
.nlist { padding-left: 1.3rem !important; margin: 0.3rem 0 0.4rem 0 !important; color: #d7d9ea; }
.nlist li { margin: 6px 0; }

.callout {
    margin: 0.6rem 0; padding: 0.55rem 0.9rem; border-radius: 10px; font-size: 0.9rem;
    background: rgba(183,163,242,.14); border: 1px solid rgba(183,163,242,.3); color: #ece7ff;
}
.codeblock {
    margin: 0.7rem 0; border: 1px solid var(--line); border-radius: 12px; background: #0b0d18; overflow: hidden;
}
.codeblock .lang { padding: 6px 14px; font-size: 0.72rem; color: var(--muted); border-bottom: 1px solid var(--line); }
.codeblock pre {
    margin: 0; padding: 0.8rem 1rem; overflow-x: auto; white-space: pre;
    font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; line-height: 1.55; color: #cfe9ff; background: transparent;
}

.ans-foot {
    display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;
    margin-top: 1rem; padding-top: 0.8rem; border-top: 1px solid var(--line); font-size: 0.78rem; color: var(--muted);
}
.grounded {
    display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 999px;
    border: 1px solid rgba(111,227,177,.45); background: rgba(111,227,177,.10); color: var(--mint); font-size: 0.74rem;
}
.err {
    margin: 0.6rem 0; padding: 0.7rem 1rem; border-radius: 12px; font-size: 0.88rem; line-height: 1.5;
    border: 1px solid rgba(255,107,129,.45); background: rgba(255,107,129,.10); color: #ffd6dc;
}

/* ---------- chat input ---------- */
[data-testid="stBottom"], [data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] {
    background: transparent !important;
}
[data-testid="stBottom"] > div { background: linear-gradient(180deg, transparent, var(--bg) 45%) !important; }
[data-testid="stChatInput"] {
    background: rgba(18,20,36,.92); border: 1px solid rgba(244,201,168,.45); border-radius: 22px;
    box-shadow: 0 0 30px rgba(244,201,168,.08);
}
[data-testid="stChatInput"] > div, [data-testid="stChatInput"] textarea {
    background: transparent !important; color: var(--text) !important; border-radius: 22px;
}
[data-testid="stChatInput"] textarea::placeholder { color: var(--muted) !important; }
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, #f3d3bd, #b8a2f0) !important; border-radius: 50%;
}
[data-testid="stChatInput"] button svg { fill: #1a1530; }

/* ---------- right panel ---------- */
[data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--line); }
button[data-baseweb="tab"] { background: transparent; }
button[data-baseweb="tab"] p { color: var(--muted); font-size: 0.85rem; font-weight: 500; }
button[data-baseweb="tab"][aria-selected="true"] p { color: #fff; font-weight: 600; }
[data-baseweb="tab-highlight"] { background: var(--peach) !important; }
[data-baseweb="tab-border"] { background: transparent !important; }

.panel-h { display: flex; align-items: center; gap: 10px; font-weight: 600; font-size: 0.9rem; margin: 0.8rem 0 0.8rem 0; }
.count { font-size: 0.72rem; padding: 2px 9px; border: 1px solid var(--line); border-radius: 999px; color: var(--muted); }
.src-item {
    display: flex; align-items: center; gap: 12px; padding: 10px 12px; margin-bottom: 8px;
    border: 1px solid var(--line); border-radius: 14px; background: var(--glass);
}
.src-item.top { border-color: rgba(244,201,168,.55); background: rgba(244,201,168,.07); }
.src-item .grow { flex: 1; min-width: 0; }
.src-item .nm { font-size: 0.85rem; font-weight: 500; word-break: break-all; }
.src-item .sub { font-size: 0.72rem; color: var(--muted); }
.src-item .chev { color: var(--muted); font-size: 1.1rem; }
.excerpt {
    position: relative; border: 1px solid var(--line); border-radius: 14px; background: var(--glass);
    padding: 0.8rem 0.9rem 0.7rem 0.9rem; margin-bottom: 10px;
}
.excerpt .mark { font-family: 'Cormorant Garamond', serif; font-size: 1.6rem; line-height: .6; color: var(--peach); }
.excerpt .tx { font-size: 0.82rem; color: #cfd1e2; line-height: 1.55; margin-top: 2px; }
.excerpt .from { font-size: 0.7rem; color: var(--muted); margin-top: 8px; }
.ctx {
    border-left: 3px solid var(--lav); background: var(--glass); border-radius: 0 12px 12px 0;
    padding: 0.7rem 0.9rem; margin-bottom: 10px;
}
.ctx .from { font-size: 0.72rem; color: var(--peach); font-weight: 600; margin-bottom: 4px; }
.ctx .tx { font-size: 0.82rem; color: #cfd1e2; line-height: 1.55; }
.ins { list-style: none; padding: 0; margin: 0; }
.ins li { display: flex; gap: 10px; align-items: baseline; font-size: 0.85rem; padding: 7px 0; border-bottom: 1px solid var(--line); }
.ins li:last-child { border-bottom: none; }
.ins .ck { color: var(--mint); }
.ins .lb { color: var(--muted); }
.empty-panel { font-size: 0.85rem; color: var(--muted); padding: 1rem 0.2rem; line-height: 1.6; }

/* ---------- alerts / spinner ---------- */
[data-testid="stAlert"] { background: rgba(111,227,177,.10); border: 1px solid rgba(111,227,177,.35); border-radius: 12px; }
[data-testid="stAlert"] * { color: var(--text) !important; }
[data-testid="stSpinner"] * { color: var(--peach) !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# helpers:start (pure functions, no Streamlit calls)
# ---------------------------------------------------------------------------
# Matches citations such as [file.pdf#chunk3], [data.csv#row12] and the
# 【file.pdf#chunk3】 form that some models produce.
_CITATION_RE = re.compile(
    r"(?:\s*,?\s*[\[【][^\]】]*#(?:chunk|row)\d+[^\]】]*[\]】])+", re.IGNORECASE
)


def extract_text(answer) -> str:
    """Turn whatever the LLM returned into plain text.

    Some models return a list of content blocks like
    [{'type': 'text', 'text': '...', 'extras': {...}}] (sometimes as a string).
    """
    if isinstance(answer, (list, tuple)):
        parts = [extract_text(item) for item in answer]
        return "\n".join(p for p in parts if p).strip()
    if isinstance(answer, dict):
        if isinstance(answer.get("text"), str):
            return answer["text"].strip()
        if "content" in answer:
            return extract_text(answer["content"])
        return ""
    if isinstance(answer, str):
        s = answer.strip()
        if s[:1] in ("[", "{"):
            for parser in (ast.literal_eval, json.loads):
                try:
                    parsed = parser(s)
                except Exception:
                    continue
                if isinstance(parsed, (list, dict)):
                    extracted = extract_text(parsed)
                    if extracted:
                        return extracted
        return s
    return str(answer or "")


def clean_answer(text: str) -> str:
    """Remove inline citations from the model's answer (they show in the Sources panel)."""
    text = _CITATION_RE.sub("", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def human_size(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{max(1, round(n / 1024))} KB"


def now_str() -> str:
    return datetime.now().strftime("%I:%M %p").lstrip("0")


def file_icon(name: str) -> str:
    ext = Path(name).suffix.lstrip(".").lower() or "file"
    cls = ext if ext in {"pdf", "txt", "csv"} else "file"
    return f'<span class="ficon {cls}">{html.escape(ext.upper()[:3])}</span>'


def group_sources(sources: list[dict] | None) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for s in sources or []:
        grouped.setdefault(s.get("source") or "Unknown", []).append(s)
    return grouped


def short_chunk(chunk_id) -> str:
    """'Handbook.pdf#chunk10' -> 'chunk10'."""
    return str(chunk_id or "unknown").split("#", 1)[-1]


def one_line(text: str, limit: int | None = None) -> str:
    """Collapse whitespace (so no blank lines can break the HTML) and escape it."""
    flat = " ".join(str(text).split())
    if limit and len(flat) > limit:
        flat = flat[:limit].rstrip() + "..."
    return html.escape(flat)


# ---- Answer structuring: Markdown -> title / intro / numbered section cards ----
_H = re.compile(r"^(#{1,4})\s+(.+)$")
_BOLD_ONLY = re.compile(r"^\*\*([^*]+?)\*\*:?$")
_NUM = re.compile(r"^(\d+)[.)]\s+(.+)$")
_BUL = re.compile(r"^[-*•]\s+(.+)$")
_CHECK_TITLES = re.compile(
    r"example|benefit|advantage|goal|consider|result|solution|strength|feature|step|pros", re.I
)
_LEAD = re.compile(
    r"^(please\s+)?(can you\s+|could you\s+)?"
    r"(explain|describe|summari[sz]e|tell me about|tell me|what is|what are|what's|define|give me|list|show me|how does|how do)\s+",
    re.I,
)


def title_from_question(question: str) -> str:
    """'What are the types of AI?' -> 'Types of AI' (max 6 words)."""
    t = _LEAD.sub("", (question or "").strip()).strip(" ?!.")
    t = re.sub(r"^(the|a|an)\s+", "", t, flags=re.I)
    words = t.split()[:6]
    if not words:
        return "Answer"
    t = " ".join(words)
    return t[0].upper() + t[1:]


def inline(text: str) -> str:
    """Escape HTML, then apply `code`, **bold** and *italic*."""
    s = html.escape(text, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![*\w])\*(?!\s)([^*]+?)(?<!\s)\*(?![*\w])", r"<em>\1</em>", s)
    return s


def parse_answer(text: str) -> dict:
    """Split Markdown into {'title', 'intro': [blocks], 'sections': [{'title', 'blocks'}]}.

    A block is ("p", text) | ("ul", [items]) | ("ol", [items]) |
    ("quote", text) | ("code", lang, code).
    """
    title = ""
    intro: list = []
    sections: list[dict] = []
    target = intro
    para: list[str] = []

    def flush_para() -> None:
        if para:
            target.append(("p", " ".join(para)))
            para.clear()

    def add_list(kind: str, item: str) -> None:
        if target and target[-1][0] == kind:
            target[-1][1].append(item)
        else:
            target.append((kind, [item]))

    def new_section(heading: str) -> None:
        nonlocal target
        section = {"title": heading, "blocks": []}
        sections.append(section)
        target = section["blocks"]

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if line.startswith("```"):
            flush_para()
            lang = line[3:].strip()
            code: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1  # closing fence
            target.append(("code", lang, "\n".join(code)))
            continue

        if not line:
            flush_para()
            continue

        m = _H.match(line)
        if m:
            flush_para()
            level, heading = len(m.group(1)), m.group(2).strip().strip("*").strip()
            if level == 1 and not title and not sections:
                title = heading
            else:
                new_section(heading)
            continue

        m = _BOLD_ONLY.match(line)
        if m:
            flush_para()
            new_section(m.group(1).strip())
            continue

        m = _NUM.match(line)
        if m:
            flush_para()
            add_list("ol", m.group(2))
            continue

        m = _BUL.match(line)
        if m:
            flush_para()
            add_list("ul", m.group(1))
            continue

        if line.startswith(">"):
            flush_para()
            target.append(("quote", line.lstrip("> ").strip()))
            continue

        para.append(line)

    flush_para()
    return {"title": title, "intro": intro, "sections": sections}


def blocks_html(blocks: list, para_class: str, list_style: str = "dots") -> str:
    out = []
    for block in blocks:
        kind = block[0]
        if kind == "p":
            out.append(f'<p class="{para_class}">{inline(block[1])}</p>')
        elif kind == "ul":
            items = "".join(f"<li>{inline(x)}</li>" for x in block[1])
            out.append(f'<ul class="blist {list_style}">{items}</ul>')
        elif kind == "ol":
            items = "".join(f"<li>{inline(x)}</li>" for x in block[1])
            out.append(f'<ol class="nlist">{items}</ol>')
        elif kind == "quote":
            out.append(f'<div class="callout">{inline(block[1])}</div>')
        elif kind == "code":
            lang = html.escape(block[1] or "code")
            # &#10; instead of real newlines: a blank line inside the code
            # would otherwise end the HTML block and break the layout.
            code = html.escape(block[2]).replace("\n", "&#10;")
            out.append(
                f'<div class="codeblock"><div class="lang">{lang}</div><pre>{code}</pre></div>'
            )
    return "".join(out)


def answer_html(msg: dict) -> str:
    """Build the HTML for one assistant message (single line, no blank lines)."""
    text = clean_answer(extract_text(msg.get("content", "")))
    sources = msg.get("sources") or []
    if not text:
        return '<div class="err">The model returned an empty reply. Please try asking again.</div>'

    parsed = parse_answer(text)
    title = parsed["title"] or (title_from_question(msg.get("question", "")) if sources else "")

    parts = ['<div class="ans">']
    if title:
        parts.append(
            f'<div class="ans-head"><span class="ans-title">{inline(title)}</span>'
            f'<span class="gen-badge">✦ From your documents</span></div>'
        )
    parts.append(blocks_html(parsed["intro"], "ans-intro"))

    for n, sec in enumerate(parsed["sections"], 1):
        style = "checks" if _CHECK_TITLES.search(sec["title"]) else "dots"
        parts.append(
            f'<div class="ans-sec"><div class="sec-head"><span class="sec-num">{n}</span>'
            f'<span class="sec-title">{inline(sec["title"])}</span></div>'
            f'<div class="sec-body">{blocks_html(sec["blocks"], "sp", style)}</div></div>'
        )

    pill = ""
    if sources:
        n_files = len(group_sources(sources))
        pill = (
            f'<span class="grounded">✓ Grounded in {len(sources)} passage'
            f'{"s" if len(sources) != 1 else ""} · {n_files} file{"s" if n_files != 1 else ""}</span>'
        )
    parts.append(f'<div class="ans-foot"><span>{pill}</span><span>{html.escape(msg.get("time", ""))}</span></div>')
    parts.append("</div>")
    return "".join(parts)


def error_html(exc: Exception) -> str:
    detail = one_line(f"{type(exc).__name__}: {exc}", limit=300)
    return (
        '<div class="err"><strong>Could not get an answer.</strong> Your message was not sent. '
        f"Check your API key, model name and rate limits, then try again.<br>{detail}</div>"
    )


def doc_list_html(files: list[tuple[str, int]]) -> str:
    if not files:
        return '<div class="empty-note">No documents indexed yet.</div>'
    return "".join(
        f'<div class="doc-item">{file_icon(name)}<div class="meta">'
        f'<div class="nm">{html.escape(name)}</div><div class="sz">{human_size(size)}</div></div>'
        f'<span class="live"></span></div>'
        for name, size in files
    )


def sources_tab_html(sources: list[dict]) -> str:
    grouped = group_sources(sources)
    items = []
    for n, (name, chunks) in enumerate(grouped.items()):
        pages = sorted({c["page"] for c in chunks if c.get("page")})
        ids = ("pages " + ", ".join(map(str, pages))) if pages else ", ".join(
            short_chunk(c.get("chunk_id")) for c in chunks
        )
        top = " top" if n == 0 else ""
        items.append(
            f'<div class="src-item{top}">{file_icon(name)}<div class="grow">'
            f'<div class="nm">{html.escape(name)}</div>'
            f'<div class="sub">{len(chunks)} passage{"s" if len(chunks) != 1 else ""} · {html.escape(ids)}</div>'
            f"</div></div>"
        )
    return f'<div class="panel-h">Sources used <span class="count">{len(sources)}</span></div>' + "".join(items)


def excerpts_tab_html(sources: list[dict]) -> str:
    items = [
        f'<div class="excerpt"><div class="mark">“</div>'
        f'<div class="tx">{one_line(s.get("text", ""), limit=320)}</div>'
        f'<div class="from">{html.escape(str(s.get("chunk_id") or "unknown"))}'
        f'{" · page " + str(s["page"]) if s.get("page") else ""}</div></div>'
        for s in sources
    ]
    return f'<div class="panel-h">Passages behind the answer <span class="count">{len(sources)}</span></div>' + "".join(items)


def insights_tab_html(sources: list[dict], total_chunks: int, n_docs: int) -> str:
    rows = []
    if sources:
        rows.append(f'<li><span class="ck">✓</span><span>Retrieved {len(sources)} relevant passages</span></li>')
        rows.append(f'<li><span class="ck">✓</span><span>From {len(group_sources(sources))} file(s)</span></li>')
        rows.append('<li><span class="ck">✓</span><span>Answer grounded in your documents</span></li>')
    else:
        rows.append('<li><span class="lb">–</span><span>No passages matched the last message</span></li>')
    rows.append(f'<li><span class="lb">·</span><span class="lb">{total_chunks} chunks indexed from {n_docs} file(s)</span></li>')
    return f'<div class="panel-h">Retrieval insights</div><ul class="ins">{"".join(rows)}</ul>'
# helpers:end


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
# The bot answers ONLY from the files uploaded in the sidebar. Every
# "Build knowledge base" REPLACES the previous knowledge base, so old files
# (or files from another session) are never mixed in.
# ---- Deployment options (set as environment variables / Streamlit secrets) ----
# RAG_FIXED_DOC  path of a PDF bundled in the repo, e.g. data/UGProspectus2025.pdf.
#                When set, the app loads it at start-up (from the saved OCR cache, so
#                no OCR happens online) and the upload box is hidden.
# APP_PASSWORD   if set, visitors must enter it before using the app (protects your
#                Groq quota on a public link).
FIXED_DOC = os.getenv("RAG_FIXED_DOC", "").strip()
APP_PASSWORD = os.getenv("APP_PASSWORD", "")

if APP_PASSWORD and not st.session_state.get("authed"):
    st.markdown(f"### {APP_NAME}")
    entered = st.text_input("Password", type="password")
    if entered:
        if entered == APP_PASSWORD:
            st.session_state.authed = True
            st.rerun()
        st.error("Wrong password.")
    st.stop()


@st.cache_resource(show_spinner="Loading the document (first start only)...")
def get_fixed_pipeline(doc_path: str):
    """One shared pipeline for all visitors, built once per server start."""
    pipeline = RAGPipeline()
    n_chunks = pipeline.load_or_build([doc_path])
    return pipeline, n_chunks


if FIXED_DOC and "pipeline" not in st.session_state:
    doc_path = Path(FIXED_DOC)
    if not doc_path.is_absolute():
        doc_path = Path(__file__).resolve().parent / doc_path
    pipeline, n_chunks = get_fixed_pipeline(str(doc_path))
    st.session_state.pipeline = pipeline
    st.session_state.indexed = True
    st.session_state.n_chunks = n_chunks
    st.session_state.doc_files = [(doc_path.name, doc_path.stat().st_size)]

if "pipeline" not in st.session_state:
    # own folder per visitor: building an index replaces the collection in its folder,
    # so a shared folder would let one visitor wipe another visitor's knowledge base
    st.session_state.pipeline = RAGPipeline(persist_dir=tempfile.mkdtemp(prefix="chroma_"))
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "n_chunks" not in st.session_state:
    st.session_state.n_chunks = 0
if "doc_files" not in st.session_state:
    st.session_state.doc_files = []


def queue_question(text: str) -> None:
    st.session_state.pending_question = text


# ---------------------------------------------------------------------------
# Sidebar: brand, upload, indexed documents
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f'<div class="brand"><span class="logo">✦</span><div><div class="name">{APP_NAME}</div>'
        '<div class="tag">Answers only from your uploaded files</div></div></div>',
        unsafe_allow_html=True,
    )
    if not FIXED_DOC:  # upload UI is hidden when a document is bundled
        st.markdown('<div class="side-title">Upload documents</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "PDF, TXT, or CSV",
            type=["pdf", "txt", "csv"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        build = st.button("Build knowledge base", disabled=not uploaded_files)

        if build:
            try:
                progress = st.progress(0.0, text="Starting...")

                def on_progress(stage: str, fraction: float) -> None:
                    progress.progress(min(max(fraction, 0.0), 1.0) * 0.95, text=stage)

                with st.spinner("Reading, chunking and indexing (a scanned PDF is OCR'd once, then cached)..."):
                    tmp_dir = Path(tempfile.mkdtemp())
                    paths = []
                    for uploaded in uploaded_files:
                        dest = tmp_dir / uploaded.name
                        dest.write_bytes(uploaded.getbuffer())
                        paths.append(str(dest))
                    n_chunks = st.session_state.pipeline.ingest(paths, on_progress=on_progress)  # replaces the old index
                progress.empty()
                st.session_state.indexed = True
                st.session_state.n_chunks = n_chunks
                st.session_state.doc_files = [(u.name, u.size) for u in uploaded_files]
                st.session_state.messages = []  # answers about old files no longer apply
                st.success(f"Indexed {n_chunks} chunks from {len(uploaded_files)} file(s).")
                for warning in st.session_state.pipeline.warnings:
                    st.warning(warning)
            except Exception as exc:  # show the cause instead of a raw traceback
                detail = one_line(f"{type(exc).__name__}: {exc}", limit=300)
                st.markdown(
                    f'<div class="err"><strong>Could not build the knowledge base.</strong><br>{detail}</div>',
                    unsafe_allow_html=True,
                )

    st.markdown('<div class="side-title">Knowledge base</div>', unsafe_allow_html=True)
    st.markdown(doc_list_html(st.session_state.doc_files), unsafe_allow_html=True)
    st.markdown(
        '<div class="quote"><span class="spark">✦</span>'
        '<div class="q">Ask better questions. Get answers you can trace.</div></div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
# Called at top level so Streamlit pins it to the bottom of the page.
question = st.chat_input("Ask a question about your documents...")
if not question:
    question = st.session_state.pop("pending_question", None)

if st.session_state.indexed:
    pill = f'<span class="pill"><span class="dot"></span>Knowledge base ready · {st.session_state.n_chunks} chunks</span>'
else:
    pill = '<span class="pill"><span class="dot off"></span>Upload a document to begin</span>'
st.markdown(f'<div class="topbar">{pill}</div>', unsafe_allow_html=True)

chat_col, panel_col = st.columns([2.1, 1], gap="large")

with chat_col:
    show_welcome = not st.session_state.messages and not question
    if show_welcome:
        hint = (
            "Your knowledge base is ready. Ask a question below or pick a suggestion."
            if st.session_state.indexed
            else "Start by uploading a document in the sidebar."
        )
        st.markdown(
            '<div class="hero"><div class="arch"></div>'
            '<div class="mtn m1"></div><div class="mtn m2"></div><div class="mtn m3"></div>'
            '<div class="hero-text"><h1><span class="spark">✦</span> Chat with your documents</h1>'
            "<p>Upload a PDF, TXT or CSV, build the knowledge base, and ask questions. "
            "Answers are copied word for word from your files, with page numbers.</p>"
            f'<div class="hint">{hint}</div></div></div>',
            unsafe_allow_html=True,
        )
        with st.container(key="chips"):
            chip_cols = st.columns(len(SUGGESTIONS))
            for col, text in zip(chip_cols, SUGGESTIONS):
                with col:
                    st.button(
                        text,
                        key=f"chip_{text}",
                        disabled=not st.session_state.indexed,
                        on_click=queue_question,
                        args=(text,),
                    )

    # conversation so far
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant", avatar=AVATAR):
                st.markdown(answer_html(message), unsafe_allow_html=True)

    # new question
    if question:
        # history is captured BEFORE the new question is added
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]
        with st.chat_message("user"):
            st.markdown(question)

        prompt = question + STRUCTURE_HINT if USE_STRUCTURE_HINT else question
        with st.chat_message("assistant", avatar=AVATAR):
            try:
                with st.spinner("Thinking..."):
                    result = st.session_state.pipeline.answer(prompt, history=history)
            except Exception as exc:
                st.markdown(error_html(exc), unsafe_allow_html=True)
            else:
                reply = {
                    "role": "assistant",
                    "content": extract_text(result.answer),
                    "sources": result.sources,
                    "question": question,
                    "time": now_str(),
                }
                st.markdown(answer_html(reply), unsafe_allow_html=True)
                st.session_state.messages.append({"role": "user", "content": question})
                st.session_state.messages.append(reply)

# ---------------------------------------------------------------------------
# Right panel: what the latest answer was based on
# ---------------------------------------------------------------------------
with panel_col:
    last_sources = next(
        (m.get("sources") or [] for m in reversed(st.session_state.messages) if m["role"] == "assistant"),
        [],
    )
    tab_sources, tab_excerpts, tab_insights = st.tabs(["Sources", "Excerpts", "Insights"])
    empty = '<div class="empty-panel">Ask a question and the passages behind the answer will appear here.</div>'
    with tab_sources:
        st.markdown(sources_tab_html(last_sources) if last_sources else empty, unsafe_allow_html=True)
    with tab_excerpts:
        st.markdown(excerpts_tab_html(last_sources) if last_sources else empty, unsafe_allow_html=True)
    with tab_insights:
        if st.session_state.messages:
            st.markdown(
                insights_tab_html(last_sources, st.session_state.n_chunks, len(st.session_state.doc_files)),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(empty, unsafe_allow_html=True)