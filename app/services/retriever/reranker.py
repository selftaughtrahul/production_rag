"""
Cross-encoder reranker.

Takes a list of retrieved documents and a query, scores every
(query, document) pair using a cross-encoder model, and returns
the top-K documents sorted by relevance score.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any

from langsmith import traceable
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class Reranker(ABC):
    """Abstract interface for document reranking."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[Any],
        top_k: int = 5,
    ) -> list[Any]:
        pass


@dataclass
class CrossEncoderReranker(Reranker):
    """
    Rerank documents using a cross-encoder model.

    The model scores every (query, chunk) pair directly instead of
    comparing embeddings, which gives more accurate relevance judgement
    at the cost of extra compute.
    """

    model: CrossEncoder

    @traceable(run_type="retriever", name="Cross-Encoder Reranker")
    def rerank(
        self,
        query: str,
        documents: list[Any],
        top_k: int = 5,
    ) -> list[Any]:
        if not documents:
            return []

        pairs = [(query, doc.text) for doc in documents]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )

        results = []
        for doc, score in ranked[:top_k]:
            # SearchResult is frozen — use replace() to add the score to metadata
            updated_metadata = {**doc.metadata, "rerank_score": float(score)}
            results.append(replace(doc, metadata=updated_metadata))

        logger.info(
            "Cross-encoder reranked %d → %d docs (top score=%.3f)",
            len(documents), len(results), float(scores.max()) if len(scores) else 0.0,
        )
        return results
