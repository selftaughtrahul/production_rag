"""
Abstract base class for all retriever implementations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Retriever(ABC):
    """
    Abstract interface for document retrieval.

    All retriever implementations (Dense, BM25, Hybrid) must implement this.
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Retrieve the most relevant documents for a query."""
        raise NotImplementedError
