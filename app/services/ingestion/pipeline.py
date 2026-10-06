"""
Ingestion pipeline — orchestrates the full document loading → chunking → embedding → storing flow.

Moved from app/pipeline/pipeline.py into app/services/ingestion/ to
keep all ingestion-related code together.
"""
from __future__ import annotations

import logging
from typing import Any

from app.core.metrics import record_ingestion
from app.services.ingestion.document_loader import ParseReport
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
            "chunks_count": len(chunks),
        }

        report = getattr(document, "report", None)
        if isinstance(report, ParseReport):
            self._record_parse(report, document_id=document_id, chunk_count=len(chunks))
            result["parser"] = report.parser
            if report.fallback_from:
                result["parser_fallback_from"] = report.fallback_from

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
