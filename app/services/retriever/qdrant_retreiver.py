"""
Hybrid retriever backed by Qdrant.

Dense (Qwen) + sparse (BM25) search and RRF fusion all run inside Qdrant
in a single query, scoped to the tenant via metadata_filter["user_id"].
"""
from __future__ import annotations

import logging
from typing import Any

from langsmith import traceable

from app.services.ingestion.embedding import EmbeddingService
from app.services.retriever.base import Retriever
from app.services.retriever.knowledge_graph import KnowledgeGraph
from app.services.vectorstore.models import SearchResult
from app.services.vectorstore.qdrant import TENANT_KEY, QdrantVectorStore

logger = logging.getLogger(__name__)


class QdrantHybridRetriever(Retriever):

    def __init__(
        self,
        embedder: EmbeddingService,
        vector_store: QdrantVectorStore,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    @traceable(run_type="retriever", name="Qdrant Hybrid Retrieval (Dense + BM25 + RRF)")
    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if not query.strip():
            return []

        if not metadata_filter or not metadata_filter.get(TENANT_KEY):
            raise ValueError("Qdrant retrieval requires metadata_filter['user_id']")

        results = self.vector_store.hybrid_search(
            query_embedding=self.embedder.embed_query(query),
            query_text=query,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

        logger.info("Qdrant hybrid search returned %d results (top_k=%d)", len(results), top_k)
        return KnowledgeGraph().rerank(query, results, top_k=top_k)
