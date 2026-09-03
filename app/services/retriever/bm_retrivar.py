from __future__ import annotations

from typing import Any

from langsmith import traceable

from app.services.retriever.retriever import Retriever
from app.services.retriever.bm25_store import BM25Store


class BM25Retriever(Retriever):
    """
    Sparse lexical retriever using BM25.
    """

    def __init__(
        self,
        bm25_store: BM25Store,
    ) -> None:
        self.bm25_store = bm25_store

    @traceable(
        run_type="retriever",
        name="BM25 Search",
    )
    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Any]:

        if not query.strip():
            return []

        return self.bm25_store.search(
            query=query,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )