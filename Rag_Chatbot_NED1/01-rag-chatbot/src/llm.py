import os
from functools import lru_cache

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", "openai/gpt-oss-120b")
EMBEDDING_MODEL = os.getenv(
    "RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


def get_chat_model(temperature: float = 0.0) -> ChatGroq:
    return ChatGroq(model=CHAT_MODEL, temperature=temperature)


@lru_cache(maxsize=1)
def get_embeddings():
    """Created once and reused: loading the model on every call wastes time and
    memory, which matters on a small server."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)