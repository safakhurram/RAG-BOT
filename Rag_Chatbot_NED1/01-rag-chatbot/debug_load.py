"""Dump the text the bot will see, page by page, so you can compare it with the PDF.

    python debug_load.py path/to/UGProspectus2025.pdf            # writes ocr_check.txt
    python debug_load.py path/to/UGProspectus2025.pdf "Self-Finance"  # only pages containing a phrase

The bot can only be as exact as this text. If a page here is wrong (columns
mixed up, numbers misread), the answer for that page will be wrong too.
"""

import sys
from pathlib import Path

from src.loaders import load_file

pdf = sys.argv[1]
phrase = sys.argv[2].lower() if len(sys.argv) > 2 else None

out = []
for d in load_file(pdf):
    if phrase and phrase not in d.page_content.lower():
        continue
    tag = " (OCR)" if d.metadata.get("ocr") else ""
    out.append(f"===== page {d.metadata['page_number']}{tag} =====\n{d.page_content}\n")

Path("ocr_check.txt").write_text("\n".join(out), encoding="utf-8")
print(f"Wrote {len(out)} pages to ocr_check.txt")
