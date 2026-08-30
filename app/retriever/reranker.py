# app/retrieval/reranker.py

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from sentence_transformers import CrossEncoder


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
    """Rerank documents using a cross-encoder model."""

    def __post_init__(self):
        self.model = CrossEncoder("BAAI/bge-reranker-v2-m3")

    def rerank(
        self,
        query: str,
        documents: list[Any],
        top_k: int = 5,
    ) -> list[Any]:

        if not documents:
            return []

        pairs = [
            (
                query,
                document.text,
            )
            for document in documents
        ]

        scores = self.model.predict(pairs)

        ranked_documents = sorted(
            zip(documents, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )

        results = []

        for document, score in ranked_documents[:top_k]:

            # Store reranker score
            document.metadata["rerank_score"] = float(score)

            results.append(document)

        return results
