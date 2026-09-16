"""Response cache keys are scoped per user and query hash."""

from __future__ import annotations

from app.services.cache.response_cache import cache_key


def test_cache_key_differs_by_user() -> None:
    question = "What is hybrid retrieval?"
    assert cache_key("user-a", question) != cache_key("user-b", question)


def test_cache_key_normalizes_question() -> None:
    assert cache_key("u1", "  Hello ") == cache_key("u1", "hello")


def test_cache_key_changes_with_question() -> None:
    assert cache_key("u1", "alpha") != cache_key("u1", "beta")
