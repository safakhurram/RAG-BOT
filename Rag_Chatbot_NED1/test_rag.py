import pytest
from src.rag import (
    MAX_HISTORY_MESSAGES,
    NO_MATCH_MESSAGE,
    NOT_INDEXED_MESSAGE,
    RAGPipeline,
)
from .fakes import FakeChatModel, FakeEmbeddings


@pytest.fixture
def pipeline(monkeypatch, tmp_path):
    # score_threshold=None: filtering is unit-tested in isolation in
    # test_vectorstore.py, so it's disabled here to keep these tests
    # deterministic rather than depending on FakeEmbeddings' (meaningless)
    # hash-based similarity scores.
    monkeypatch.setattr("src.vectorstore.get_embeddings", lambda: FakeEmbeddings())
    monkeypatch.setattr("src.rag.get_chat_model", lambda: FakeChatModel())
    return RAGPipeline(
        persist_dir=str(tmp_path / "chroma"), score_threshold=None, strict=False
    )


def test_answer_before_ingest_does_not_call_the_llm(pipeline):
    result = pipeline.answer("anything")

    assert result.answer == NOT_INDEXED_MESSAGE
    assert result.sources == []
    # the chat client is never even constructed — proves no API call, and
    # that RAGPipeline() never requires an API key just to be instantiated
    assert pipeline._chat_model is None


def test_ingest_and_answer_returns_cited_sources(pipeline, tmp_path):
    doc = tmp_path / "notes.txt"
    doc.write_text("The capital of France is Paris. Paris is known for the Eiffel Tower.")
    n_chunks = pipeline.ingest([str(doc)])
    assert n_chunks >= 1

    result = pipeline.answer("What is the capital of France?")

    assert result.answer == "FAKE ANSWER"
    assert len(result.sources) > 0
    assert all(s["source"] == "notes.txt" for s in result.sources)
    assert all(s["chunk_id"] for s in result.sources)


def test_system_prompt_includes_retrieved_context(pipeline, tmp_path):
    doc = tmp_path / "notes.txt"
    doc.write_text("Zylphoria is a fictional planet with three moons.")
    pipeline.ingest([str(doc)])

    pipeline.answer("Tell me about Zylphoria")

    system_message = pipeline._chat_model.received_messages[0].content
    assert "Zylphoria" in system_message


def test_ingest_raises_on_empty_content(pipeline, tmp_path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("")

    with pytest.raises(ValueError):
        pipeline.ingest([str(empty_file)])


def test_no_documents_indexed_message_mentions_uploading():
    assert "upload" in NOT_INDEXED_MESSAGE.lower()


def test_no_match_message_does_not_claim_an_answer():
    assert "don't have enough information" in NO_MATCH_MESSAGE.lower()

def test_unmatched_question_still_goes_to_the_llm_with_persona(monkeypatch, tmp_path):
    monkeypatch.setattr("src.vectorstore.get_embeddings", lambda: FakeEmbeddings())
    monkeypatch.setattr("src.rag.get_chat_model", lambda: FakeChatModel())
    # threshold above the maximum possible score => nothing is ever retrieved
    pipeline = RAGPipeline(
        persist_dir=str(tmp_path / "chroma"), score_threshold=2.0, strict=False
    )
    doc = tmp_path / "notes.txt"
    doc.write_text("The capital of France is Paris.")
    pipeline.ingest([str(doc)])

    result = pipeline.answer("you are useless")

    assert result.answer == "FAKE ANSWER"  # came from the model, not a canned string
    assert result.sources == []
    system_message = pipeline._chat_model.received_messages[0].content
    assert "No relevant passages" in system_message


def test_system_prompt_contains_de_escalation_rules(pipeline, tmp_path):
    doc = tmp_path / "notes.txt"
    doc.write_text("Some content.")
    pipeline.ingest([str(doc)])

    pipeline.answer("hello")

    system_message = pipeline._chat_model.received_messages[0].content.lower()
    assert "calm" in system_message
    assert "out-of-scope" in system_message


def test_history_is_sent_to_the_model(pipeline, tmp_path):
    doc = tmp_path / "notes.txt"
    doc.write_text("The capital of France is Paris.")
    pipeline.ingest([str(doc)])
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]

    pipeline.answer("and the second one?", history=history)

    contents = [m.content for m in pipeline._chat_model.received_messages]
    assert contents[1:] == ["hi", "hello", "and the second one?"]


def test_history_is_trimmed_to_the_limit(pipeline, tmp_path):
    doc = tmp_path / "notes.txt"
    doc.write_text("The capital of France is Paris.")
    pipeline.ingest([str(doc)])
    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
        for i in range(20)
    ]

    pipeline.answer("latest question", history=history)

    # system message + trimmed history + current question
    assert len(pipeline._chat_model.received_messages) == 1 + MAX_HISTORY_MESSAGES + 1