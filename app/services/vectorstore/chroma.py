from __future__ import annotations
from typing import Any
import chromadb
from .base import VectorStore
from .models import EmbeddedChunk, SearchResult


class ChromaVectorStore(VectorStore):

    def __init__(
        self,
        persist_directory: str | None,
        collection_name: str,
        embedding_dimension: int,
        *,
        host: str | None = None,
        port: int | None = None,
    ) -> None:

        if embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be greater than 0")

        self.embedding_dimension = embedding_dimension

        if host:
            self.client = chromadb.HttpClient(host=host, port=port or 8000)
        else:
            self.client = chromadb.PersistentClient(path=persist_directory or "data/chroma_db")

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
            },
        )

    def add(
        self,
        chunks: list[EmbeddedChunk],
    ) -> None:

        if not chunks:
            return

        self._validate_chunks(chunks)

        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=[chunk.embedding for chunk in chunks],
            metadatas=[self._sanitize_metadata(chunk.metadata) for chunk in chunks],
        )

    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:

        self._validate_embedding(query_embedding)

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
        }

        if metadata_filter:

            query_kwargs["where"] = self._build_filter(metadata_filter)

        results = self.collection.query(**query_kwargs)

        ids = results["ids"][0]
        documents = results["documents"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]

        return [
            SearchResult(
                chunk_id=chunk_id,
                text=document,
                score=1.0 / (1.0 + distance),
                metadata=metadata or {},
            )
            for chunk_id, document, distance, metadata in zip(
                ids,
                documents,
                distances,
                metadatas,
            )
        ]

    def delete(
        self,
        chunk_ids: list[str],
    ) -> None:

        if not chunk_ids:
            return

        self.collection.delete(ids=chunk_ids)

    def count(self) -> int:

        return self.collection.count()

    def health_check(self) -> bool:

        try:

            self.client.list_collections()

            return True

        except Exception:

            return False

    @staticmethod
    def _sanitize_metadata(
        metadata: dict[str, Any],
    ) -> dict[str, Any]:

        sanitized = {}

        for key, value in metadata.items():

            if value is None:
                continue

            if isinstance(
                value,
                (str, int, float, bool),
            ):
                sanitized[key] = value

            else:
                sanitized[key] = str(value)

        return sanitized

    @staticmethod
    def _build_filter(
        metadata_filter: dict[str, Any],
    ) -> dict[str, Any]:

        return {key: {"$eq": value} for key, value in metadata_filter.items()}

    def _validate_chunks(
        self,
        chunks: list[EmbeddedChunk],
    ) -> None:

        for chunk in chunks:
            self._validate_embedding(chunk.embedding)

    def _validate_embedding(
        self,
        embedding: list[float],
    ) -> None:

        if len(embedding) != self.embedding_dimension:

            raise ValueError(
                "Embedding dimension mismatch: "
                f"expected {self.embedding_dimension}, "
                f"got {len(embedding)}"
            )
