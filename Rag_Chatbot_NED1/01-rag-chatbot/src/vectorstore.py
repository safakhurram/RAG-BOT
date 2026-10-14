"""Persistence and retrieval over a Chroma vector store, plus keyword search.

Retrieval uses cosine similarity explicitly (rather than Chroma's raw L2
default) so relevance scores land in a well-understood 0-1 range, which is
what makes `filter_by_relevance` below a meaningful, tunable cutoff.

Two fixes for prospectus-style documents:

* `build_vectorstore` REPLACES the collection. Previously every click on
  "Build knowledge base" appended the same chunks again (the old database held
  every prospectus chunk 3 times, plus chunks from unrelated PDFs uploaded in
  earlier sessions), so top-k was filled with duplicates and stale documents.
* `hybrid_search` merges the vector results with BM25 keyword results.
  Embeddings are weak at exact tokens such as "SF-6", "Rs.", programme codes
  and people's names; keyword search is strong at exactly those.
"""

import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .llm import get_embeddings
from .sections import search_text

COLLECTION_NAME = "documents"
COLLECTION_METADATA = {"hnsw:space": "cosine"}


def build_vectorstore(chunks: list[Document], persist_dir: str | Path) -> Chroma:
    old = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_dir),
        embedding_function=get_embeddings(),
        collection_metadata=COLLECTION_METADATA,
    )
    old.delete_collection()
    # embed "section + text" so a table chunk is found by its programme name
    indexed = [Document(page_content=search_text(c), metadata=dict(c.metadata)) for c in chunks]
    return Chroma.from_documents(
        documents=indexed,
        embedding=get_embeddings(),
        ids=[c.metadata["chunk_id"] for c in chunks],
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_dir),
        collection_metadata=COLLECTION_METADATA,
    )


def load_vectorstore(persist_dir: str | Path) -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_dir),
        embedding_function=get_embeddings(),
        collection_metadata=COLLECTION_METADATA,
    )


def filter_by_relevance(
    results: list[tuple[Document, float]], threshold: float | None
) -> list[Document]:
    """Drop retrieved chunks whose relevance score falls below `threshold`.
    `threshold=None` disables filtering and returns everything."""
    if threshold is None:
        return [doc for doc, _ in results]
    return [doc for doc, score in results if score >= threshold]


def search_relevant(
    vectorstore: Chroma, query: str, k: int, score_threshold: float | None
) -> list[Document]:
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    return filter_by_relevance(results, score_threshold)


_TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")
_STOP = {
    "the", "a", "an", "of", "for", "to", "in", "on", "and", "or", "is", "are",
    "what", "which", "who", "how", "much", "many", "tell", "me", "about", "please",
    "do", "does", "can", "i", "my", "at", "by", "with", "from", "be", "it", "this",
}


def _stem(token: str) -> str:
    """Very light plural folding: departments -> department, fees -> fee."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOP]


_ORDINALS = {
    "1": "first", "1st": "first", "2": "second", "2nd": "second",
    "3": "third", "3rd": "third", "4": "fourth", "4th": "fourth", "final": "fourth",
}


def expand_query(query: str) -> str:
    """'3 year' / '3rd year' -> also 'third year', as the prospectus writes it."""
    extra = [
        _ORDINALS[m.group(1).lower()]
        for m in re.finditer(r"\b(\d(?:st|nd|rd|th)?|final)\s*(?:-\s*)?year", query, re.I)
        if m.group(1).lower() in _ORDINALS
    ]
    return query + (" " + " ".join(f"{w} year" for w in extra) if extra else "")


class KeywordIndex:
    """BM25 over the same chunks that are in the vector store."""

    def __init__(self, chunks: list[Document]):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(search_text(c)) or ["_"] for c in chunks])

    def search(self, query: str, k: int) -> list[Document]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i] for i in ranked[:k] if scores[i] > 0]


def hybrid_search(
    vectorstore: Chroma,
    keyword_index: KeywordIndex | None,
    query: str,
    k: int,
    score_threshold: float | None,
) -> list[Document]:
    """Reciprocal-rank fusion of vector and keyword results, best first.
    Results are the ORIGINAL chunks (without the section prefix used for search)."""
    query = expand_query(query)
    vector_hits = search_relevant(vectorstore, query, k, score_threshold)
    keyword_hits = keyword_index.search(query, k) if keyword_index else []
    if keyword_index:
        originals = {c.metadata.get("chunk_id"): c for c in keyword_index.chunks}
        vector_hits = [originals.get(d.metadata.get("chunk_id"), d) for d in vector_hits]

    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    for hits in (vector_hits, keyword_hits):
        for rank, doc in enumerate(hits):
            key = doc.metadata.get("chunk_id") or doc.page_content
            docs.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ordered[:k]]
