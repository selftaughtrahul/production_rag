"""Context dedupe, character budget, source numbering, and empty-context refusal."""

from __future__ import annotations

from app.prompts.specialist_prompts import RAG_GENERATION_SYSTEM_V2, RAG_NO_EVIDENCE_ANSWER
from app.services.rag.nodes import RAGNodes
from app.services.retriever.context import ContextBuilder
from app.services.vectorstore.models import SearchResult


def _chunk(chunk_id: str, text: str, score: float, **meta) -> SearchResult:
    metadata = {"filename": "handbook.pdf", **meta}
    return SearchResult(chunk_id=chunk_id, text=text, score=score, metadata=metadata)


def test_identical_text_kept_once() -> None:
    builder = ContextBuilder(max_chars=10_000)
    built = builder.build([
        _chunk("a", "Refunds are issued within 14 days.", 0.9),
        _chunk("b", "Refunds  are issued within 14 days. ", 0.8),
    ])
    assert len(built.sources) == 1
    assert built.sources[0].chunk_id == "a"
    assert built.text.count("Refunds are issued") == 1


def test_same_chunk_id_kept_once() -> None:
    builder = ContextBuilder(max_chars=10_000)
    built = builder.build([
        _chunk("a", "Alpha text.", 0.9),
        _chunk("a", "Different text, same id.", 0.5),
    ])
    assert [s.chunk_id for s in built.sources] == ["a"]


def test_contained_overlap_dropped() -> None:
    builder = ContextBuilder(max_chars=10_000)
    long = "Section 2. Refunds are issued within 14 days of the request being approved by finance."
    short = "Refunds are issued within 14 days"
    built = builder.build([
        _chunk("short", short, 0.95),
        _chunk("long", long, 0.9),
    ])
    assert [s.chunk_id for s in built.sources] == ["long"]


def test_near_duplicate_dropped_by_jaccard() -> None:
    builder = ContextBuilder(max_chars=10_000, near_duplicate_threshold=0.8)
    base = "one two three four five six seven eight nine ten"
    built = builder.build([
        _chunk("a", base, 0.9),
        _chunk("b", "one two three four five six seven eight nine eleven ten", 0.7),
        _chunk("c", "totally unrelated words about something else entirely here", 0.6),
    ])
    assert [s.chunk_id for s in built.sources] == ["a", "c"]


def test_budget_stops_adding_chunks() -> None:
    # Each block is "[n] handbook.pdf\n" (17 chars) + 50 chars; blocks join with "\n\n".
    builder = ContextBuilder(max_chars=150)
    built = builder.build([
        _chunk("a", "A" * 50, 0.9),
        _chunk("b", "B" * 50, 0.8),
        _chunk("c", "C" * 50, 0.7),
    ])
    assert [s.chunk_id for s in built.sources] == ["a", "b"]
    assert len(built.text) <= 150
    assert "CCCC" not in built.text


def test_first_chunk_always_kept_even_over_budget() -> None:
    builder = ContextBuilder(max_chars=10)
    built = builder.build([_chunk("a", "A" * 50, 0.9)])
    assert [s.chunk_id for s in built.sources] == ["a"]


def test_sources_numbered_in_rerank_order_with_page() -> None:
    builder = ContextBuilder(max_chars=10_000)
    built = builder.build([
        _chunk("x", "First block.", 0.5, page=3),
        _chunk("y", "Second block.", 0.9, filename="policy.md"),
    ])
    assert [(s.index, s.filename, s.page) for s in built.sources] == [
        (1, "handbook.pdf", 3),
        (2, "policy.md", None),
    ]
    assert built.text.startswith("[1] handbook.pdf, page 3\nFirst block.")
    assert "\n\n[2] policy.md\nSecond block." in built.text
    assert built.sources[0].to_dict() == {
        "index": 1,
        "filename": "handbook.pdf",
        "page": 3,
        "chunk_id": "x",
    }


def test_empty_input_gives_empty_context() -> None:
    built = ContextBuilder(max_chars=100).build([])
    assert built.text == ""
    assert built.sources == []


class _NoCallLLM:
    def generate(self, **_: object) -> str:
        raise AssertionError("LLM must not be called when context is empty")


def test_generate_refuses_without_model_call_on_empty_context() -> None:
    nodes = RAGNodes(retriever=None, reranker=None, context_builder=None, llm=_NoCallLLM())
    update = nodes.generate({"question": "What is the refund window?", "context": "   "})
    assert update["answer"] == RAG_NO_EVIDENCE_ANSWER
    assert update["chat_history"][1].content == RAG_NO_EVIDENCE_ANSWER


def test_build_context_node_stores_sources() -> None:
    nodes = RAGNodes(
        retriever=None,
        reranker=None,
        context_builder=ContextBuilder(max_chars=10_000),
        llm=None,
    )
    update = nodes.build_context({"documents": [_chunk("a", "Alpha.", 0.9)]})
    assert update["sources"] == [
        {"index": 1, "filename": "handbook.pdf", "page": None, "chunk_id": "a"}
    ]
    assert update["context"].startswith("[1] handbook.pdf\nAlpha.")


def test_prompt_contains_refusal_and_cite_rule() -> None:
    assert RAG_NO_EVIDENCE_ANSWER in RAG_GENERATION_SYSTEM_V2
    assert "[n]" in RAG_GENERATION_SYSTEM_V2
