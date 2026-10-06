from __future__ import annotations

import logging
import uuid
from typing import Any

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    HnswConfigDiff,
    KeywordIndexParams,
    MatchValue,
    Modifier,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from .base import VectorStore
from .models import EmbeddedChunk, SearchResult

logger = logging.getLogger(__name__)

TENANT_KEY = "user_id"


class QdrantVectorStore(VectorStore):
    """
    Qdrant-backed multi-tenant hybrid vector store.

    Each point holds a named dense vector (from the embedding model) and a
    named sparse BM25 vector (from fastembed). Hybrid search fuses both
    with RRF inside Qdrant.

    Multi-tenancy: ``user_id`` is stored at the top level of the payload and
    indexed with ``is_tenant=True``. Every other metadata field lives under
    ``payload["metadata"]``.
    """

    def __init__(
        self,
        collection_name: str,
        embedding_dimension: int,
        url: str,
        api_key: str | None = None,
        *,
        dense_vector_name: str = "dense_vector",
        sparse_vector_name: str = "sparse_vector",
        sparse_model: str = "Qdrant/bm25",
        sparse_cache_dir: str | None = None,
        batch_size: int = 128,
    ) -> None:

        if embedding_dimension <= 0:
            raise ValueError(
                "embedding_dimension must be greater than 0"
            )
        if not collection_name:
            raise ValueError("collection_name must be set")
        if not url:
            raise ValueError("Qdrant url must be set")

        self.embedding_dimension = embedding_dimension
        self.collection_name = collection_name
        self.dense_vector_name = dense_vector_name
        self.sparse_vector_name = sparse_vector_name
        self.batch_size = batch_size

        self.client = QdrantClient(url=url, api_key=api_key)

        logger.info("Loading sparse model %s", sparse_model)
        self.sparse_model = SparseTextEmbedding(
            model_name=sparse_model,
            cache_dir=sparse_cache_dir,
        )

        self._ensure_collection()

    # =========================================================
    # COLLECTION MANAGEMENT
    # =========================================================

    def _ensure_collection(self) -> None:
        """
        Create the collection if missing, otherwise validate that it has
        the expected named vectors and dimension.
        """

        if not self.client.collection_exists(self.collection_name):

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    self.dense_vector_name: VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    self.sparse_vector_name: SparseVectorParams(
                        modifier=Modifier.IDF,
                    ),
                },
                # Per-tenant HNSW graphs instead of one global graph.
                hnsw_config=HnswConfigDiff(m=0, payload_m=16),
            )

        info = self.client.get_collection(
            collection_name=self.collection_name
        )
        self._ensure_payload_indexes(info.payload_schema or {})

        params = info.config.params

        vectors = params.vectors
        if not isinstance(vectors, dict) or self.dense_vector_name not in vectors:
            raise ValueError(
                f"Qdrant collection '{self.collection_name}' has no named "
                f"dense vector '{self.dense_vector_name}'"
            )

        existing_dimension = vectors[self.dense_vector_name].size
        if existing_dimension != self.embedding_dimension:
            raise ValueError(
                "Qdrant collection embedding dimension mismatch: "
                f"collection={existing_dimension}, "
                f"provider={self.embedding_dimension}"
            )

        sparse = params.sparse_vectors or {}
        if self.sparse_vector_name not in sparse:
            raise ValueError(
                f"Qdrant collection '{self.collection_name}' has no sparse "
                f"vector '{self.sparse_vector_name}'"
            )

        if sparse[self.sparse_vector_name].modifier != Modifier.IDF:
            logger.warning(
                "Sparse vector '%s' has no IDF modifier; BM25 scores will "
                "ignore term rarity",
                self.sparse_vector_name,
            )

    def _ensure_payload_indexes(self, payload_schema: dict[str, Any]) -> None:
        """
        Qdrant Cloud (strict mode) rejects filters on unindexed fields,
        so index the tenant key and document_id.
        """

        indexes = {
            TENANT_KEY: KeywordIndexParams(type="keyword", is_tenant=True),
            "metadata.document_id": KeywordIndexParams(type="keyword"),
            "metadata.tags": KeywordIndexParams(type="keyword"),
        }

        for field_name, schema in indexes.items():
            if field_name not in payload_schema:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema,
                    wait=True,
                )

    # =========================================================
    # ADD / UPSERT
    # =========================================================

    def add(
        self,
        chunks: list[EmbeddedChunk],
    ) -> None:

        if not chunks:
            return

        self._validate_chunks(chunks)

        for start in range(0, len(chunks), self.batch_size):

            batch = chunks[start:start + self.batch_size]

            sparse_embeddings = self.sparse_model.embed(
                [chunk.text for chunk in batch]
            )

            points: list[PointStruct] = []

            for chunk, sparse in zip(batch, sparse_embeddings, strict=True):

                metadata = self._sanitize_metadata(chunk.metadata)
                user_id = metadata.get(TENANT_KEY)

                if not user_id:
                    raise ValueError(
                        f"chunk {chunk.chunk_id} is missing metadata.user_id"
                    )

                points.append(
                    PointStruct(
                        id=self._point_id(chunk.chunk_id),
                        vector={
                            self.dense_vector_name: chunk.embedding,
                            self.sparse_vector_name: SparseVector(
                                indices=sparse.indices.tolist(),
                                values=sparse.values.tolist(),
                            ),
                        },
                        payload={
                            TENANT_KEY: user_id,
                            "chunk_id": chunk.chunk_id,
                            "text": chunk.text,
                            "metadata": metadata,
                        },
                    )
                )

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

    # =========================================================
    # DENSE VECTOR SEARCH
    # =========================================================

    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:

        self._validate_embedding(query_embedding)

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        points = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            using=self.dense_vector_name,
            query_filter=self._build_filter(metadata_filter),
            limit=top_k,
            with_payload=True,
        ).points

        return [self._to_search_result(point) for point in points]

    # =========================================================
    # HYBRID SEARCH (DENSE + BM25, RRF)
    # =========================================================

    def hybrid_search(
        self,
        query_embedding: list[float],
        query_text: str,
        *,
        top_k: int = 5,
        prefetch_limit: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Run dense and sparse searches inside Qdrant and fuse them with RRF.
        """

        self._validate_embedding(query_embedding)

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        prefetch_limit = prefetch_limit or top_k * 2
        query_filter = self._build_filter(metadata_filter)

        sparse = next(iter(self.sparse_model.query_embed(query_text)))

        points = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query=query_embedding,
                    using=self.dense_vector_name,
                    filter=query_filter,
                    limit=prefetch_limit,
                ),
                Prefetch(
                    query=SparseVector(
                        indices=sparse.indices.tolist(),
                        values=sparse.values.tolist(),
                    ),
                    using=self.sparse_vector_name,
                    filter=query_filter,
                    limit=prefetch_limit,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        ).points

        return [self._to_search_result(point) for point in points]

    # =========================================================
    # DELETE CHUNKS
    # =========================================================

    def delete(
        self,
        chunk_ids: list[str],
    ) -> None:

        if not chunk_ids:
            return

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=[self._point_id(cid) for cid in chunk_ids],
            wait=True,
        )

    # =========================================================
    # DELETE DOCUMENT
    # =========================================================

    def delete_document(
        self,
        document_id: str,
        user_id: str | None = None,
    ) -> int:
        """
        Delete all chunks belonging to a document.

        Pass user_id to restrict the delete to that tenant.
        """

        document_filter = self._document_filter(
            document_id,
            {TENANT_KEY: user_id} if user_id else None,
        )

        deleted = self.client.count(
            collection_name=self.collection_name,
            count_filter=document_filter,
            exact=True,
        ).count

        if deleted:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=document_filter,
                wait=True,
            )

        return int(deleted)

    # =========================================================
    # COUNT DOCUMENT CHUNKS
    # =========================================================

    def count_document_chunks(
        self,
        document_id: str,
        metadata_filter: dict | None = None,
    ) -> int:
        """
        Count chunks belonging to a document.

        Optional metadata_filter can be used for ownership
        verification.
        """

        result = self.client.count(
            collection_name=self.collection_name,
            count_filter=self._document_filter(document_id, metadata_filter),
            exact=True,
        )

        return int(result.count)

    # =========================================================
    # DOCUMENT EXISTS
    # =========================================================

    def document_exists(
        self,
        document_id: str,
        metadata_filter: dict | None = None,
    ) -> bool:

        return self.count_document_chunks(document_id, metadata_filter) > 0

    # =========================================================
    # GET DOCUMENT CHUNKS
    # =========================================================

    def get_document_chunks(
        self,
        document_id: str,
        metadata_filter: dict | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return all chunks belonging to a document.
        """

        query_filter = self._document_filter(document_id, metadata_filter)

        results: list[dict[str, Any]] = []
        offset = None

        while True:

            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=1_000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                result = self._to_search_result(point)
                results.append(
                    {
                        "chunk_id": result.chunk_id,
                        "text": result.text,
                        "metadata": result.metadata,
                    }
                )

            if offset is None:
                return results

    # =========================================================
    # COUNT
    # =========================================================

    def count(self) -> int:

        result = self.client.count(
            collection_name=self.collection_name,
            exact=True,
        )

        return int(result.count)

    # =========================================================
    # HEALTH CHECK
    # =========================================================

    def health_check(self) -> bool:

        try:
            self.client.get_collections()
            return True

        except Exception:
            return False

    # =========================================================
    # HELPERS
    # =========================================================

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        """Qdrant IDs must be UUIDs or ints; derive a stable UUID."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    @staticmethod
    def _to_search_result(point: Any) -> SearchResult:

        payload = point.payload or {}

        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        return SearchResult(
            chunk_id=str(payload.get("chunk_id", point.id)),
            text=str(payload.get("text", "")),
            score=float(getattr(point, "score", 0.0) or 0.0),
            metadata=metadata,
        )

    @staticmethod
    def _sanitize_metadata(
        metadata: dict[str, Any],
    ) -> dict[str, Any]:

        sanitized: dict[str, Any] = {}

        for key, value in metadata.items():

            if value is None:
                continue

            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, (list, tuple)):
                # Keep keyword arrays (e.g. tags) so Qdrant can match any element.
                sanitized[key] = [
                    item if isinstance(item, (str, int, float, bool)) else str(item)
                    for item in value
                    if item is not None
                ]
            else:
                sanitized[key] = str(value)

        return sanitized

    @staticmethod
    def _build_filter(
        metadata_filter: dict[str, Any] | None,
    ) -> Filter | None:
        """
        user_id maps to the top-level tenant field; every other key
        maps to payload["metadata"][key].

        A list value (e.g. tags) becomes one condition per element, so every
        listed value must be present on the point (AND semantics).
        """

        if not metadata_filter:
            return None

        conditions: list[FieldCondition] = []
        for key, value in metadata_filter.items():
            field = key if key == TENANT_KEY else f"metadata.{key}"
            values = value if isinstance(value, (list, tuple, set)) else [value]
            for item in values:
                if item is None:
                    continue
                conditions.append(FieldCondition(key=field, match=MatchValue(value=item)))

        return Filter(must=conditions) if conditions else None

    def _document_filter(
        self,
        document_id: str,
        metadata_filter: dict[str, Any] | None = None,
    ) -> Filter:

        if not document_id.strip():
            raise ValueError(
                "document_id cannot be empty"
            )

        return self._build_filter(
            {**(metadata_filter or {}), "document_id": document_id}
        )

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
