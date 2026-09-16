"""Cheap supervisor routing hints."""

from __future__ import annotations

from app.agent.multi_agent.route import next_agent


def test_greeting_goes_to_general() -> None:
    assert next_agent("hello", []) == "general_agent"


def test_document_hint_goes_to_rag() -> None:
    assert next_agent("what did I upload in the pdf", []) == "rag_agent"


def test_sql_hint_goes_to_sql() -> None:
    assert next_agent("how many rows are in the table", []) == "sql_agent"


def test_web_hint_goes_to_web() -> None:
    assert next_agent("latest news on tsla", []) == "web_agent"


def test_ambiguous_hints_defer_to_llm() -> None:
    assert next_agent("sql policy in the pdf", []) is None


def test_second_hop_finishes() -> None:
    assert next_agent("anything", ["rag_agent"]) == "FINISH"


def test_empty_query_is_general() -> None:
    assert next_agent("   ", []) == "general_agent"
