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

_SKIP_PREPARED = frozenset(
    {"source", "user_id", "document_id", "chunk_id", "chunk_index", "tags"}
)

logger = logging.getLogger(__name__)


def prepare_document_metadata(
    raw: dict[str, Any],
    *,
    filename: str,
    content_type: str,
    file_size: int | None,
    image_count: int,
) -> dict[str, str]:
    """Turn parser facts into short string metadata stored with the document."""
    merged: dict[str, Any] = {
        **raw,
        "filename": filename,
        "content_type": content_type,
        "image_count": image_count,
    }
    if file_size is not None:
        merged["file_size"] = file_size

    prepared: dict[str, str] = {}
    for key, value in merged.items():
        if key in _SKIP_PREPARED or value is None or isinstance(value, (list, dict)):
            continue
        text = str(value).strip()
        if not text:
            continue
        prepared[str(key)[:64]] = text[:128]
        if len(prepared) >= 30:
            break
    return prepared


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
            content_hash: SHA-256 of the upload, stored on every chunk.
            file_size: Byte length of the upload.

        Returns:
            dict with document_id, filename, user_id, metadata, chunks_count.
        """
        document = self.loader.load(source)
        image_labels = extract_embedded_images(
            Path(source),
            Path(source).parent / f"{document_id}_images",
        )
        if image_labels:
            document.text = f"{document.text.rstrip()}\n\n" + "\n".join(image_labels)

        extension = Path(filename).suffix.lower()
        content_type = content_type_for(extension)
        prepared = prepare_document_metadata(
            document.metadata,
            filename=filename,
            content_type=content_type,
            file_size=file_size,
            image_count=len(image_labels),
        )
        document_metadata: dict[str, Any] = {
            **document.metadata,
            **prepared,
            "document_id": document_id,
            "filename": filename,
            "user_id": user_id,
            "content_type": content_type,
            "image_count": len(image_labels),
        }
        if content_hash:
            document_metadata["content_hash"] = content_hash
        if file_size is not None:
            document_metadata["file_size"] = file_size

        cleaned_text = self.cleaner.clean(document.text)
        chunks = self.chunker.chunk(cleaned_text, metadata=document_metadata)

        for index, chunk in enumerate(chunks):
            chunk.metadata["document_id"] = document_id
            chunk.metadata["filename"] = filename
            chunk.metadata["user_id"] = user_id
            chunk.metadata["chunk_index"] = index
            chunk.metadata["chunk_id"] = f"{document_id}_chunk_{index}"
            chunk.metadata.pop("tags", None)

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
            "metadata": prepared,
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
