"""Hybrid retriever: empty query, tenant filter, RRF fusion."""

from __future__ import annotations

from typing import Any

from app.services.retriever.hybrid import HybridRetriever
from app.services.vectorstore.models import SearchResult


class _StubRetriever:
    def __init__(self, docs: list[SearchResult]) -> None:
        self.docs = docs
        self.last_filter: dict[str, Any] | None = None

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        self.last_filter = metadata_filter
        return self.docs[:top_k]


def _doc(chunk_id: str, text: str) -> SearchResult:
    return SearchResult(chunk_id=chunk_id, text=text, score=1.0, metadata={})


def test_blank_query_returns_no_docs() -> None:
    hybrid = HybridRetriever(_StubRetriever([_doc("a", "alpha")]), _StubRetriever([]))
    assert hybrid.retrieve("   ", top_k=5) == []


def test_metadata_filter_is_passed_to_both_retrievers() -> None:
    dense = _StubRetriever([_doc("a", "alpha")])
    bm25 = _StubRetriever([_doc("b", "beta")])
    hybrid = HybridRetriever(dense, bm25)
    tenant = {"user_id": "u-1"}
    hybrid.retrieve("alpha", top_k=5, metadata_filter=tenant)
    assert dense.last_filter == tenant
    assert bm25.last_filter == tenant


def test_rrf_prefers_docs_in_both_lists() -> None:
    both = _doc("both", "appears in dense and bm25")
    dense_only = _doc("dense", "dense only")
    bm25_only = _doc("bm25", "bm25 only")
    hybrid = HybridRetriever(
        _StubRetriever([dense_only, both]),
        _StubRetriever([bm25_only, both]),
    )
    fused = hybrid.retrieve("query", top_k=3)
    assert [doc.chunk_id for doc in fused][0] == "both"
    assert fused[0].metadata.get("rrf_score", 0) > 0
