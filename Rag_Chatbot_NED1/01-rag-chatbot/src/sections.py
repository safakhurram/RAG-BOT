"""Tag every chunk with the section of the document it belongs to.

A chunk taken from the middle of a table or a staff list often doesn't name
its subject. The third-year table of Civil Engineering, for example, is a
chunk that starts with "THIRD YEAR" and never says "Civil". Searching for
"courses in 3rd year of civil" can't find it.

So we walk through the document, track the latest numbered heading
("3.1.1 Civil Engineering", "2.2.15 Department of Computer Science ...") and
the latest year heading ("THIRD YEAR"), and store them on each chunk as
metadata["section"]. The section is added to the text used for SEARCH and
shown to the model next to each block. It is never part of the quotable text,
so answers stay word for word from the PDF.
"""

import re

from langchain_core.documents import Document

# "3.1.1 | Civil Engineering", "2.2.15 Department of ...", "2.2.29(i) Department ..."
# Table-of-contents lines end with a page number and are ignored.
_HEADING = re.compile(
    r"^(\d+(?:\.\d+)+(?:\s*\([ivx]+\))?)\.?\s*\|?\s*([A-Za-z][^|]{2,100}?)\s*$"
)
_YEAR = re.compile(r"^(FIRST|SECOND|THIRD|FOURTH|FINAL)\s+YEAR\b.*$", re.I)


def _line_sections(pages: list[Document]) -> list[list[str]]:
    """For each page, the section label in effect at each of its lines."""
    heading, year, pending_number = "", "", None
    result = []
    for page in pages:
        labels = []
        for line in page.page_content.splitlines():
            stripped = line.strip()
            is_toc = bool(re.search(r"(\||\s)\d{1,3}\s*$", stripped))
            if pending_number and stripped and not is_toc and re.match(r"^[A-Za-z]", stripped):
                # "3.1.2" on one line, its title on the next
                heading, year = f"{pending_number} {stripped.split('|')[0].strip()}", ""
                pending_number = None
            else:
                pending_number = None
                m = _HEADING.match(stripped)
                if re.fullmatch(r"\d+(?:\.\d+)+(?:\s*\([ivx]+\))?\.?", stripped):
                    pending_number = stripped.rstrip(".")
                elif m and not is_toc:
                    heading, year = f"{m.group(1)} {m.group(2).strip()}", ""
                elif _YEAR.match(stripped):
                    year = stripped.split("|")[0].strip().upper()
            labels.append(heading + (f" > {year}" if year and heading else year))
        result.append(labels)
    return result


def add_sections(pages: list[Document], chunks: list[Document]) -> None:
    """Set chunk.metadata["section"] from the page the chunk was cut from."""
    by_key: dict[tuple, tuple[Document, list[str]]] = {}
    for page, labels in zip(pages, _line_sections(pages)):
        key = (page.metadata.get("source"), page.metadata.get("page"))
        by_key[key] = (page, labels)

    for chunk in chunks:
        key = (chunk.metadata.get("source"), chunk.metadata.get("page"))
        if key not in by_key:
            continue
        page, labels = by_key[key]
        pos = page.page_content.find(chunk.page_content[:80])
        first = page.page_content.count("\n", 0, pos) if pos >= 0 else 0
        last = first + chunk.page_content.count("\n")
        # the section at the chunk's start plus any section that begins inside it
        found: list[str] = []
        for label in labels[min(first, len(labels) - 1) : last + 1]:
            if label and label not in found:
                found.append(label)
        if found:
            chunk.metadata["section"] = " / ".join(found[:4])


def search_text(chunk: Document) -> str:
    """Text used for search: section label + chunk text."""
    section = chunk.metadata.get("section")
    return f"{section}\n{chunk.page_content}" if section else chunk.page_content
