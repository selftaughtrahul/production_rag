from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .models import EmbeddedChunk, SearchResult


class VectorStore(ABC):
    """
    Provider-independent vector store interface.

    Implementations:
        - PGVectorStore
        - QdrantVectorStore
        - ChromaVectorStore
    """

    @abstractmethod
    def add(
        self,
        chunks: list[EmbeddedChunk],
    ) -> None:
        """
        Insert or update chunks.
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Search for semantically similar chunks.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(
        self,
        chunk_ids: list[str],
    ) -> None:
        """
        Delete chunks by ID.
        """
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """
        Return number of stored chunks.
        """
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> bool:
        """
        Check whether the vector store is available.
        """
        raise NotImplementedError
