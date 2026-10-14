import sys

from src.loaders import load_file

docs = load_file(sys.argv[1])
print(f"Done: {len(docs)} pages with text")