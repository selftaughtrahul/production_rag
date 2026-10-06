"""
Ingestion pipeline — orchestrates the full document loading → chunking → embedding → storing flow.

Moved from app/pipeline/pipeline.py into app/services/ingestion/ to
keep all ingestion-related code together.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.core.metrics import record_ingestion
from app.services.ingestion.document_loader import (
    ParseReport,
    content_type_for,
    extract_embedded_images,
)
from app.services.vectorstore.base import VectorStore

_RESERVED_METADATA = frozenset(
    {
        "document_id",
        "filename",
        "user_id",
        "tags",
        "chunk_index",
        "chunk_id",
        "source",
        "content_hash",
        "file_size",
        "content_type",
        "image_count",
    }
)

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

    def ingest(
        self,
        source: str,
        document_id: str,
        filename: str,
        user_id: str,
        tags: list[str] | None = None,
        custom_metadata: dict[str, str] | None = None,
        content_hash: str | None = None,
        file_size: int | None = None,
    ) -> dict[str, Any]:
        """
        Run the full ingestion pipeline for a single document.

        Args:
            source:      Path to the temporary file on disk.
            document_id: UUID assigned to this document.
            filename:    Original filename uploaded by the user.
            user_id:     ID of the authenticated user — stored on every chunk
                         to enable per-user metadata filtering at retrieval time.
            tags:        User-supplied labels copied onto every chunk as
                         ``metadata.tags`` for query-time filtering.
            custom_metadata: User key/value chips copied onto every chunk.
            content_hash: SHA-256 of the upload, stored on every chunk.
            file_size: Byte length of the upload.

        Returns:
            dict with document_id, filename, user_id, chunks_count.
        """
        document = self.loader.load(source)
        tags = list(tags or [])
        image_labels = extract_embedded_images(
            Path(source),
            Path(source).parent / f"{document_id}_images",
        )
        if image_labels:
            document.text = f"{document.text.rstrip()}\n\n" + "\n".join(image_labels)

        extension = Path(filename).suffix.lower()
        document_metadata: dict[str, Any] = {
            **document.metadata,
            "document_id": document_id,
            "filename": filename,
            "user_id": user_id,
            "tags": tags,
            "content_type": content_type_for(extension),
            "image_count": len(image_labels),
        }
        if content_hash:
            document_metadata["content_hash"] = content_hash
        if file_size is not None:
            document_metadata["file_size"] = file_size
        for key, value in (custom_metadata or {}).items():
            if key not in _RESERVED_METADATA and value:
                document_metadata[key] = value

        cleaned_text = self.cleaner.clean(document.text)
        chunks = self.chunker.chunk(cleaned_text, metadata=document_metadata)

        # System fields are written last so user tags can never override them.
        for index, chunk in enumerate(chunks):
            chunk.metadata["tags"] = tags
            chunk.metadata["document_id"] = document_id
            chunk.metadata["filename"] = filename
            chunk.metadata["user_id"] = user_id
            chunk.metadata["chunk_index"] = index
            chunk.metadata["chunk_id"] = f"{document_id}_chunk_{index}"

        stored = 0
        for batch in self.embedder.embed_chunks(chunks):
            self.vector_store.add(batch)
            stored += len(batch)
        if stored != len(chunks):
            raise RuntimeError("Stored chunk count does not match the document")

        logger.info(
            "Ingested document '%s' (id=%s, user=%s, chunks=%d)",
            filename, document_id, user_id, len(chunks),
        )

        result: dict[str, Any] = {
            "document_id": document_id,
            "filename": filename,
            "user_id": user_id,
            "tags": tags,
            "chunks_count": len(chunks),
        }

        report = getattr(document, "report", None)
        if isinstance(report, ParseReport):
            self._record_parse(report, document_id=document_id, chunk_count=len(chunks))
            result["parser"] = report.parser
            result["ocr_used"] = report.ocr_used
            result["table_count"] = report.table_count
            if report.fallback_from:
                result["parser_fallback_from"] = report.fallback_from
        result["image_count"] = len(image_labels)

        return result

    @staticmethod
    def _record_parse(report: ParseReport, *, document_id: str, chunk_count: int) -> None:
        """Push parser facts to Prometheus; never fail the ingest."""
        logger.info(
            "Parsed document id=%s parser=%s type=%s pages=%s elements=%d tables=%d "
            "chunks=%d failed_pages=%d ocr=%s fallback_from=%s seconds=%.3f",
            document_id, report.parser, report.document_type, report.page_count,
            report.element_count, report.table_count, chunk_count,
            len(report.failed_pages), report.ocr_used, report.fallback_from,
            report.processing_seconds,
        )
        try:
            record_ingestion(
                parser=report.parser,
                document_type=report.document_type,
                page_count=report.page_count,
                element_count=report.element_count,
                table_count=report.table_count,
                chunk_count=chunk_count,
                failed_pages=len(report.failed_pages),
                ocr_used=report.ocr_used,
                fell_back=report.fallback_from is not None,
                processing_seconds=report.processing_seconds,
            )
        except Exception:  # noqa: BLE001 - metrics must not break ingestion
            logger.warning("Failed to record ingestion metrics for %s", document_id)
