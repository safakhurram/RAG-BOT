"""Check that a quote the model "copied" is really in the source text.

`locate_quote` finds the quote inside the chunk, tolerating only differences
in whitespace/line breaks, letter case and tiny copying slips, and returns the
text exactly as it appears in the SOURCE. The model's own version of the text
is never shown to the user, so even a small paraphrase ("Rs 51,930" vs
"Rs. 51,930/-") is corrected back to the PDF wording, and anything the model
made up is rejected because it can't be found.
"""

import re

from rapidfuzz import fuzz

MIN_SCORE = 92  # 0-100 similarity required for a fuzzy match
MIN_QUOTE_CHARS = 4


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace and lowercase, remembering each output char's
    index in the original text."""
    out: list[str] = []
    index: list[int] = []
    in_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if out and not in_space:
                out.append(" ")
                index.append(i)
            in_space = True
            continue
        in_space = False
        low = ch.lower()
        if len(low) != 1:
            low = ch
        out.append(low)
        index.append(i)
    if out and out[-1] == " ":
        out.pop()
        index.pop()
    return "".join(out), index


def _expand_to_lines(text: str, start: int, end: int) -> tuple[int, int]:
    """Always return whole source lines, so a quote can never cut off part of
    an amount, a name or a sentence (e.g. "Rs. 336,630" without its "/-")."""
    start = text.rfind("\n", 0, start) + 1
    nl = text.find("\n", end)
    end = len(text) if nl == -1 else nl
    return start, end


def locate_quote(quote: str, source: str, min_score: float = MIN_SCORE) -> str | None:
    """Return the exact source text matching `quote`, or None if it isn't there."""
    q = normalize(quote)
    if len(q) < MIN_QUOTE_CHARS:
        return None
    src_norm, index = _normalize_with_map(source)
    if not src_norm or len(q) > len(src_norm) + 5:
        return None

    pos = src_norm.find(q)
    if pos != -1:
        s, e = pos, pos + len(q)
    else:
        align = fuzz.partial_ratio_alignment(q, src_norm)
        if align is None or align.score < min_score:
            return None
        s, e = align.dest_start, align.dest_end
        # the matched span must cover (almost) the whole quote
        if (e - s) < 0.9 * len(q):
            return None

    start, end = index[s], index[e - 1] + 1
    start, end = _expand_to_lines(source, start, end)
    result = source[start:end].strip()
    return result or None


def quote_already_shown(quote: str, previous_answer: str) -> bool:
    """True if this exact passage was in the previous answer ("tell me more")."""
    if not previous_answer:
        return False
    prev = re.sub(r"^\s*-\s*", "", previous_answer, flags=re.M)
    return normalize(quote) in normalize(prev)
