"""See what the bot retrieves for a question, and whether the OCR text even contains a word.

  python debug_query.py "Who is the chairperson of telecom engineering" --grep telecom --grep chair

A file picker opens so you can choose the PDF. (Or pass --pdf "path/to/file.pdf".)
"""
import argparse

from src.rag import RAGPipeline
from src.vectorstore import expand_query, hybrid_search

ap = argparse.ArgumentParser()
ap.add_argument("question")
ap.add_argument("--pdf", help="optional; if omitted a file picker opens")
ap.add_argument("--grep", action="append", default=[], help="word to look for in ALL chunks")
args = ap.parse_args()

pdf = args.pdf
if not pdf:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    pdf = filedialog.askopenfilename(title="Choose the PDF", filetypes=[("PDF files", "*.pdf")])
    if not pdf:
        raise SystemExit("No file chosen.")

p = RAGPipeline()
print(f"{p.load_or_build([pdf])} chunks indexed")
print("expanded query:", expand_query(args.question), "\n")

print("=== 1. WORD SEARCH IN THE OCR TEXT (is it there at all?) ===")
for word in args.grep:
    found = [c for c in p._keyword_index.chunks if word.lower() in c.page_content.lower()]
    print(f"'{word}': {len(found)} chunk(s)")
    for c in found[:8]:
        i = c.page_content.lower().find(word.lower())
        snippet = " ".join(c.page_content[max(0, i - 80): i + 160].split())
        print(f"  {c.metadata['chunk_id']} p.{c.metadata.get('page_number')}: ...{snippet}...")

print("\n=== 2. WHAT THE MODEL WOULD BE GIVEN ===")
hits = p._with_next_chunks(
    hybrid_search(p._vectorstore, p._keyword_index, args.question, p.top_k, p.score_threshold)
)
for d in hits:
    print(f"--- {d.metadata['chunk_id']} p.{d.metadata.get('page_number')} section={d.metadata.get('section')}")
    print(d.page_content[:400], "\n")

print("=== 3. FINAL ANSWER ===")
print(p.answer(args.question).answer)