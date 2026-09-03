"""
Dense vector retriever using ChromaDB.

Embeds the query with a HuggingFace model and performs
approximate nearest-neighbour search in the vector store.
"""
from __future__ import annotations

import logging
from typing import Any

from langsmith import traceable

from app.services.ingestion.embedding import EmbeddingService
from app.services.retriever.base import Retriever
from app.services.vectorstore.chroma import ChromaVectorStore

logger = logging.getLogger(__name__)


class DenseRetriever(Retriever):
    """
    Semantic vector search using ChromaDB.

    Converts the query into an embedding and finds the closest
    chunks in the vector store via cosine similarity.
    """

    def __init__(
        self,
        embedder: EmbeddingService,
        vector_store: ChromaVectorStore,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    @traceable(run_type="retriever", name="Dense Vector Search")
    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Any]:
        if not query.strip():
            return []

        query_embedding = self.embedder.embed_query(query)

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

        logger.info("Dense search returned %d results (top_k=%d)", len(results), top_k)
        return results


# Backwards-compatible alias — import DenseRetriever or RetrieverService, both work
RetrieverService = DenseRetriever
