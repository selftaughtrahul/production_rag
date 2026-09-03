"""
BM25 sparse keyword retriever backed by SQLite FTS5.
"""
from __future__ import annotations

import logging
from typing import Any

from langsmith import traceable

from app.services.retriever.base import Retriever
from app.services.retriever.bm25_store import BM25Store

logger = logging.getLogger(__name__)


class BM25Retriever(Retriever):
    """
    Lexical keyword retriever using BM25 ranking (SQLite FTS5).

    Complements the DenseRetriever — strong for exact keyword matches
    where semantic search may miss.
    """

    def __init__(self, bm25_store: BM25Store) -> None:
        self.bm25_store = bm25_store

    @traceable(run_type="retriever", name="BM25 Keyword Search")
    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Any]:
        if not query.strip():
            return []

        results = self.bm25_store.search(
            query=query,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

        logger.info("BM25 search returned %d results (top_k=%d)", len(results), top_k)
        return results
