"""
Hybrid retriever — Dense Vector Search + BM25 Keyword Search fused with RRF.

Pipeline:
    Dense (Chroma)  ──┐
                      ├─→  RRF Fusion  →  top-K results
    BM25 (SQLite)   ──┘
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from langsmith import traceable

from app.services.retriever.base import Retriever
from app.services.retriever.dense import DenseRetriever
from app.services.retriever.bm25 import BM25Retriever

logger = logging.getLogger(__name__)


class HybridRetriever(Retriever):
    """
    Combines dense vector search and BM25 keyword search.

    Results from both retrievers are merged using Reciprocal Rank Fusion (RRF),
    which boosts documents that appear highly ranked in both lists.
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever

    @traceable(run_type="retriever", name="Hybrid Retrieval (Dense + BM25 + RRF)")
    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Any]:
        if not query.strip():
            return []

        # 1. Dense vector search
        dense_results = self.dense_retriever.retrieve(
            query=query,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

        # 2. BM25 keyword search
        sparse_results = self.bm25_retriever.retrieve(
            query=query,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

        logger.info(
            "Hybrid retrieval: dense=%d, bm25=%d docs",
            len(dense_results), len(sparse_results),
        )

        # 3. Fuse with Reciprocal Rank Fusion
        fused = self._rrf_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            top_k=top_k,
        )

        logger.info("After RRF fusion: %d docs returned", len(fused))
        return fused

    def _rrf_fusion(
        self,
        dense_results: list[Any],
        sparse_results: list[Any],
        top_k: int,
        k: int = 60,
    ) -> list[Any]:
        """
        Reciprocal Rank Fusion.

        Score formula: sum(1 / (k + rank)) across all result lists.
        Documents appearing high in multiple lists get a boosted score.

        Args:
            dense_results:  Ranked list from the dense retriever.
            sparse_results: Ranked list from the BM25 retriever.
            top_k:          Number of results to return.
            k:              RRF constant (default 60 — standard value from the paper).
        """
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, Any] = {}

        for result_list in (dense_results, sparse_results):
            for rank, doc in enumerate(result_list, start=1):
                cid = doc.chunk_id
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
                doc_map[cid] = doc

        ranked_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)

        fused: list[Any] = []
        for cid in ranked_ids[:top_k]:
            doc = doc_map[cid]
            updated_metadata = {**doc.metadata, "rrf_score": round(rrf_scores[cid], 6)}
            fused.append(replace(doc, metadata=updated_metadata))

        return fused
