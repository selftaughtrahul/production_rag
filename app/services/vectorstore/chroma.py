from __future__ import annotations

from typing import Any

import chromadb

from .base import (
    VectorStore,
    EmbeddedChunk,
    SearchResult,
)


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
            self.client = chromadb.HttpClient(
                host=host,
                port=port or 8000,
            )

        else:
            self.client = chromadb.PersistentClient(
                path=persist_directory or "data/chroma_db"
            )

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

        ids = results.get("ids", [])[0]
        distances = results.get("distances", [])[0]
        documents = results.get("documents", [])[0]
        metadatas = results.get("metadatas", [])[0]

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

    # =========================================================
    # DELETE CHUNKS
    # =========================================================

    def delete(
        self,
        chunk_ids: list[str],
    ) -> None:

        if not chunk_ids:
            return

        self.collection.delete(ids=chunk_ids)

    def delete_document(
        self,
        document_id: str,
    ) -> int:
        """
        Delete all chunks/vectors belonging to a document.

        Example:

            document_id = "abc123"

        This removes:

            abc123_chunk_0
            abc123_chunk_1
            abc123_chunk_2
            ...

        based on the document_id stored in metadata.

        Returns:
            Number of chunks that existed before deletion.
        """

        if not document_id.strip():

            raise ValueError("document_id cannot be empty")

        existing = self.collection.get(
            where={"document_id": document_id},
            include=[
                "metadatas",
            ],
        )

        chunk_ids = existing.get(
            "ids",
            [],
        )

        if not chunk_ids:
            return 0

        self.collection.delete(where={"document_id": document_id})

        return len(chunk_ids)

    def count_document_chunks(
        self,
        document_id: str,
        metadata_filter: dict | None = None,
    ) -> int:
        """
        Count how many chunks belong to a document.

        Args:
            document_id:     The document UUID.
            metadata_filter: Optional extra filters (e.g. {"user_id": "..."})
                             used for ownership verification before delete.
        """
        if not document_id.strip():
            raise ValueError("document_id cannot be empty")

        where: dict = {"document_id": document_id}

        if metadata_filter:
            # Combine document_id filter with extra filters using $and
            conditions = [{"document_id": {"$eq": document_id}}]
            for key, value in metadata_filter.items():
                conditions.append({key: {"$eq": value}})
            where = {"$and": conditions}

        result = self.collection.get(
            where=where,
            include=[],
        )

        return len(result.get("ids", []))


    def document_exists(
        self,
        document_id: str,
    ) -> bool:
        """
        Check whether at least one vector/chunk
        exists for a document.
        """

        if not document_id.strip():

            raise ValueError("document_id cannot be empty")

        result = self.collection.get(
            where={"document_id": document_id},
            include=[],
        )

        return bool(
            result.get(
                "ids",
                [],
            )
        )

    def get_document_chunks(
        self,
        document_id: str,
    ) -> list[dict[str, Any]]:
        """
        Return all chunks belonging to a document.

        Useful for debugging and document management.
        """

        if not document_id.strip():

            raise ValueError("document_id cannot be empty")

        result = self.collection.get(
            where={"document_id": document_id},
            include=[
                "documents",
                "metadatas",
            ],
        )

        ids = result.get(
            "ids",
            [],
        )

        documents = result.get(
            "documents",
            [],
        )

        metadatas = result.get(
            "metadatas",
            [],
        )

        return [
            {
                "chunk_id": chunk_id,
                "text": text,
                "metadata": metadata or {},
            }
            for chunk_id, text, metadata in zip(
                ids,
                documents,
                metadatas,
            )
        ]

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

        sanitized: dict[str, Any] = {}

        for key, value in metadata.items():

            if value is None:
                continue

            if isinstance(
                value,
                (
                    str,
                    int,
                    float,
                    bool,
                ),
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
