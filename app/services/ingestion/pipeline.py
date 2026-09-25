"""
Ingestion pipeline — orchestrates the full document loading → chunking → embedding → storing flow.

Moved from app/pipeline/pipeline.py into app/services/ingestion/ to
keep all ingestion-related code together.
"""
from __future__ import annotations

import logging
from typing import Any

from app.services.vectorstore.base import VectorStore

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Orchestrates the complete document ingestion pipeline.

    Steps:
        1. Load document from source path
        2. Clean text
        3. Chunk text
        4. Embed chunks
        5. Upsert to the Qdrant vector store (dense + BM25 sparse vectors)
    """

    def __init__(self, loader, cleaner, chunker, embedder, vector_store: VectorStore) -> None:
        self.loader = loader
        self.cleaner = cleaner
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

    def ingest(self,source: str,document_id: str,filename: str,user_id: str) -> dict[str, Any]:
        """
        Run the full ingestion pipeline for a single document.

        Args:
            source:      Path to the temporary file on disk.
            document_id: UUID assigned to this document.
            filename:    Original filename uploaded by the user.
            user_id:     ID of the authenticated user — stored on every chunk
                         to enable per-user metadata filtering at retrieval time.

        Returns:
            dict with document_id, filename, user_id, chunks_count.
        """
        document = self.loader.load(source)

        document_metadata = {
            **document.metadata,
            "document_id": document_id,
            "filename": filename,
            "user_id": user_id,
        }

        cleaned_text = self.cleaner.clean(document.text)
        chunks = self.chunker.chunk(cleaned_text, metadata=document_metadata)

        for index, chunk in enumerate(chunks):
            chunk.metadata["document_id"] = document_id
            chunk.metadata["filename"] = filename
            chunk.metadata["user_id"] = user_id
            chunk.metadata["chunk_index"] = index
            chunk.metadata["chunk_id"] = f"{document_id}_chunk_{index}"

        embedded_chunks = self.embedder.embed_chunks(chunks)
        self.vector_store.add(embedded_chunks)

        logger.info(
            "Ingested document '%s' (id=%s, user=%s, chunks=%d)",
            filename, document_id, user_id, len(chunks),
        )

        return {
            "document_id": document_id,
            "filename": filename,
            "user_id": user_id,
            "chunks_count": len(chunks),
        }
