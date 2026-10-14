# RAG Chatbot for Documents (PDF / TXT / CSV)

## Strict "only what is in the PDF" mode (default)

Why answers used to differ from the prospectus, and what changed:

| Problem | Fix |
|---|---|
| OCR glued English words together ("DeanFacultyofCivil...") because `rapidocr_onnxruntime` uses a Chinese recogniser | `rapidocr` v3 with its bundled PP-OCRv6 models keeps the spaces (`src/loaders.py`) |
| Two-column pages were read line by line across both columns, so fee amounts sat next to labels from the other column | OCR boxes are re-ordered column by column; table rows are kept as `label \| amount` (`src/ocr_layout.py`) |
| Header/footer lines on every page polluted every chunk | lines repeated on 40%+ of pages are removed |
| Each "Build knowledge base" click appended the same chunks again (the old DB had every chunk 3 times, plus chunks from other PDFs) | the collection is replaced on every build (`src/vectorstore.py`) |
| Exact terms (SF-6, names, programme codes) were missed by embeddings | hybrid search: vector + BM25 keyword search |
| The model wrote its own answer, so it could paraphrase, merge or round | the model only copies passages; code checks each one against the source text and shows the PDF's own wording, whole lines only, with page numbers. Anything not found in the source is dropped (`src/rag.py`, `src/verify.py`) |
| Previous bot replies were sent back to the model and could be repeated as facts | in strict mode the answer model never sees earlier replies |

Set `RAG_STRICT_MODE=0` in `.env` to go back to free-written (generative) answers.

### Upload: the bot answers only from what you upload

Upload a PDF/TXT/CSV in the sidebar and click **Build knowledge base**. Each build
REPLACES the previous knowledge base (and clears the chat), so the bot never mixes
in older files. The OCR of the NED prospectus is already included in `ocr_cache/`,
so uploading `UGProspectus2025.pdf` does not need to OCR it again (the cache is
matched by file content). A new scanned PDF is OCR'd once and then cached.

Other improvements for this PDF:

* Every chunk is tagged with its section (e.g. `3.1.1 Civil Engineering > THIRD YEAR`)
  so a table chunk is found by its programme and year. The tag is used for search and
  shown to the model, but it is never part of the quoted text.
* "3 year" / "3rd year" also searches "third year"; plurals are folded (departments -> department).
* The chunk after each top match is also given to the model, so tables cut across two
  chunks arrive whole.
* Citations show the page number printed on the sheet and the PDF viewer page,
  e.g. `page 93 (PDF page 100)`.

### Run

```bash
pip install -r requirements.txt
rm -rf chroma_db                 # Windows: rmdir /s /q chroma_db   (old, duplicated index)
streamlit run app.py             # run from this folder (it uses ocr_cache/)
```

Building the knowledge base for the prospectus takes about a minute (embedding). To inspect the exact text the bot sees:
`python debug_load.py path/to/UGProspectus2025.pdf` -> `ocr_check.txt`.

### Known limits

* Strict mode only quotes. It does not count or calculate, so "How many departments
  are there?" gets "not enough information": the prospectus lists the departments but
  never states their number.
* A few characters are misread by OCR (e.g. "AI" read as "Al", "Drawing-II" as
  "Drawing-Il"). The bot shows the OCR text exactly.
* Rotated/graphic tables (e.g. seat distribution) OCR poorly; check them in
  `ocr_check.txt`.

---

## General project notes (original README)

Chat with your own PDFs, text files, and CSVs. Every answer cites exactly
which chunk of which file it came from — this is what separates a real RAG
system from "paste the file into ChatGPT."

### What it does

- Upload PDF, TXT, and CSV files
- Loads → chunks → embeds → indexes them in a local vector store
- Ask questions in a chat UI; answers are generated **only** from retrieved
  content, with inline `[source#chunk_id]` citations
- If nothing relevant was retrieved, it says so instead of guessing

### How it works

```
 upload           load            chunk              embed + index         retrieve + answer
┌──────────┐    ┌─────────┐    ┌───────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│ file.pdf │───▶│ loaders │───▶│   chunking    │───▶│    vectorstore    │───▶│         rag          │
│ file.txt │    │  .py    │    │     .py       │    │  (Chroma, cosine)  │    │ retrieve top-k above │
│ file.csv │    └─────────┘    │ prose: token- │    └──────────────────┘    │ relevance threshold, │
└──────────┘                   │ aware splitter│                            │ then ask the LLM using│
                                │ CSV row: kept │                            │ ONLY that context     │
                                │ atomic        │                            └─────────────────────┘
                                └───────────────┘
```

Two implementation details worth calling out (this is what most "chat with
your PDF" tutorials skip):

1. **Chunking is type-aware.** Prose (PDF/TXT) is split by *token* count with
   overlap (`src/chunking.py`, via tiktoken) — token count is what actually
   determines context-window usage, not character count. CSV rows are kept
   as one chunk each, since splitting a row by length would sever a record's
   fields from each other for no benefit.
2. **Retrieval has a relevance floor.** Answers only use chunks whose cosine
   similarity to the question clears `score_threshold` (default `0.25`,
   `src/rag.py`). Below that, the pipeline says it doesn't know rather than
   forcing an answer out of whatever ranked highest.

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt   # includes requirements.txt + pytest

cp .env.example .env
# then edit .env and set OPENAI_API_KEY
```

### Run it

```bash
streamlit run app.py
```

(The bundled prospectus build ignores `data/sample/`.) The original demo used the sample files in `data/sample/` (a company FAQ, an
employee handbook PDF, and an employee roster CSV — see
`data/generate_sample_data.py` for how they were generated) and ask
something like:

- "How many vacation days do employees get?"
- "Which employees work in Quality Assurance?"
- "What is Northwind Robotics' stock ticker symbol?" (tests the "I don't know" path — it's not in the docs)

### Run the tests

```bash
pytest
```

16 tests, all offline — no API key required. They cover file loading, the
type-aware chunking logic, the relevance-score filter, and the RAG pipeline's
control flow (citation assembly, the "nothing indexed yet" and "nothing
relevant found" paths), using fake embeddings/chat model stand-ins
(`tests/fakes.py`) so nothing hits the network.

### Run the eval (optional, costs a few cents)

```bash
python eval/run_eval.py
```

This is a separate, small evaluation harness — not part of `pytest` — that
runs 7 golden questions against the bundled sample data and reports two
numbers:

- **Retrieval hit-rate**: did the chunk actually containing the answer get
  retrieved in the top-k?
- **Answer quality**: a second LLM call grades whether the generated answer
  actually satisfies each question's rubric (`eval/golden_set.py`).

This is the difference between "it looks like it works" and having a number
you can point to and improve.

### Project layout

```
app.py                     Streamlit UI
src/
  loaders.py                PDF/TXT/CSV → LangChain Documents
  chunking.py                type-aware chunking (token-based prose, atomic CSV rows)
  vectorstore.py             Chroma persistence + relevance-score filtering
  rag.py                     the pipeline: ingest() and answer()
  llm.py                     the ONLY place that talks to OpenAI — swap providers here
data/sample/                bundled demo files (generated by generate_sample_data.py)
eval/                       golden Q&A set + evaluation harness (not part of pytest)
tests/                      offline unit tests (fakes.py mocks the LLM/embeddings)
```

### Swapping the LLM provider

Every LLM/embedding call goes through the two functions in `src/llm.py`.
To run against a local model instead of OpenAI (e.g. via Ollama), that's the
only file you need to change — nothing in `rag.py`, `vectorstore.py`, or
`app.py` references OpenAI directly.

### Where this is intentionally scoped down

This is the "core scope, fully working" version, not the maximal one. Known
directions to extend it (good material for "what would you improve?" in an
interview):

- Re-ranking retrieved chunks with a cross-encoder before generation
- Table-aware PDF parsing (tables currently flatten to plain text)
- Multi-hop questions that need synthesizing facts across several chunks
- Conversation memory across turns — see the companion "Conversational RAG"
  project, which builds directly on this one

### Known limitation

`langchain-community` (used for the PDF/TXT/CSV loaders) prints a
deprecation warning on import — LangChain is migrating loaders to standalone
packages. It still works correctly; this is just upstream churn to be aware
of, not a bug in this project.
