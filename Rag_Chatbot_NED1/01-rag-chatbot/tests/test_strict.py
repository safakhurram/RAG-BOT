"""Strict mode: only text that is literally in the document reaches the user."""

import json

import pytest

from src.rag import (
    GREETING_MESSAGE,
    NO_MATCH_MESSAGE,
    NOTHING_NEW_MESSAGE,
    OUT_OF_SCOPE_MESSAGE,
    RAGPipeline,
)
from src.verify import locate_quote

from .fakes import FakeChatModel, FakeEmbeddings

FEE_TEXT = (
    "8.1 FEE AND CHARGES (NON-REFUNDABLE)\n"
    "a) Admission Fee to be paid once at the time of Admission\n"
    "For locals\n"
    "All BE, BS & B.Arch except BS (CT) Rs. 51,930/-\n"
    "BS (CT) Programmes Rs. 56,390/-\n"
    "For Foreigners Rs. 336,630/-\n"
)


def make_pipeline(monkeypatch, tmp_path, reply: dict | str):
    content = reply if isinstance(reply, str) else json.dumps(reply)
    monkeypatch.setattr("src.vectorstore.get_embeddings", lambda: FakeEmbeddings())
    monkeypatch.setattr("src.rag.get_chat_model", lambda: FakeChatModel(content))
    pipeline = RAGPipeline(persist_dir=str(tmp_path / "chroma"), score_threshold=None, strict=True)
    doc = tmp_path / "prospectus.txt"
    doc.write_text(FEE_TEXT)
    pipeline.ingest([str(doc)])
    return pipeline


def test_paraphrased_quote_is_replaced_by_exact_source_text(monkeypatch, tmp_path):
    # model drops the "/-" and changes spacing/case: the user still sees the PDF wording
    reply = {
        "type": "answer",
        "quotes": [{"id": "prospectus.txt#chunk0", "text": "for foreigners rs. 336,630"}],
    }
    result = make_pipeline(monkeypatch, tmp_path, reply).answer("Fee for foreigners?")

    assert result.answer.startswith("For Foreigners Rs. 336,630/-")
    assert result.sources and result.sources[0]["chunk_id"] == "prospectus.txt#chunk0"


def test_invented_quote_is_rejected(monkeypatch, tmp_path):
    reply = {
        "type": "answer",
        "quotes": [{"id": "prospectus.txt#chunk0", "text": "For Foreigners Rs. 400,000/- per year"}],
    }
    result = make_pipeline(monkeypatch, tmp_path, reply).answer("Fee for foreigners?")

    assert result.answer == NO_MATCH_MESSAGE
    assert result.sources == []


def test_wrong_block_label_still_verifies_against_retrieved_text(monkeypatch, tmp_path):
    reply = {"type": "answer", "quotes": [{"id": "nonexistent#chunk9", "text": "BS (CT) Programmes Rs. 56,390/-"}]}
    result = make_pipeline(monkeypatch, tmp_path, reply).answer("BS CT admission fee?")

    assert "BS (CT) Programmes Rs. 56,390/-" in result.answer


def test_not_json_falls_back_to_no_match(monkeypatch, tmp_path):
    result = make_pipeline(monkeypatch, tmp_path, "The fee is about 50k.").answer("fee?")
    assert result.answer == NO_MATCH_MESSAGE


@pytest.mark.parametrize(
    "kind,expected", [("greeting", GREETING_MESSAGE), ("out_of_scope", OUT_OF_SCOPE_MESSAGE)]
)
def test_non_answers_are_fixed_sentences(monkeypatch, tmp_path, kind, expected):
    result = make_pipeline(monkeypatch, tmp_path, {"type": kind, "quotes": []}).answer("hi")
    assert result.answer == expected


def test_history_is_not_sent_to_the_answer_model(monkeypatch, tmp_path):
    reply = {"type": "not_found", "quotes": []}
    pipeline = make_pipeline(monkeypatch, tmp_path, reply)
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "made-up fact"}]

    pipeline.answer("and for foreigners?", history=history)

    sent = " ".join(m.content for m in pipeline._chat_model.received_messages)
    assert "made-up fact" not in sent


def test_tell_me_more_does_not_repeat_the_same_passage(monkeypatch, tmp_path):
    reply = {"type": "answer", "quotes": [{"id": "prospectus.txt#chunk0", "text": "For Foreigners Rs. 336,630/-"}]}
    pipeline = make_pipeline(monkeypatch, tmp_path, reply)
    first = pipeline.answer("Fee for foreigners?")
    history = [
        {"role": "user", "content": "Fee for foreigners?"},
        {"role": "assistant", "content": first.answer},
    ]
    assert pipeline.answer("tell me more", history=history).answer == NOTHING_NEW_MESSAGE


def test_locate_quote_returns_whole_lines():
    src = "8.3 SECURITY DEPOSIT\nSecurity deposit of Rs. 15,500/- is included in the admission fee.\nNext line"
    assert (
        locate_quote("deposit of Rs. 15,500", src)
        == "Security deposit of Rs. 15,500/- is included in the admission fee."
    )


def test_locate_quote_tolerates_ocr_line_breaks():
    src = "Security deposit of Rs. 15,500/- is included in the admission\nfee."
    assert locate_quote("included in the admission fee.", src) == src
