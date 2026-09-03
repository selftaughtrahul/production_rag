"""
Builds the context string passed to the LLM from retrieved chunks.
"""
from __future__ import annotations

import logging

from langsmith import traceable

from app.services.vectorstore.models import SearchResult

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Converts retrieved chunks into a single formatted context string
    that the LLM can read.
    """

    @traceable(run_type="chain", name="Build Context String")
    def build(self, chunks: list[SearchResult]) -> str:
        if not chunks:
            return ""

        parts = []
        for index, chunk in enumerate(chunks, start=1):
            source = chunk.metadata.get("filename") or chunk.metadata.get("source", "")
            source_info = f" (Source: {source})" if source else ""
            parts.append(f"[Document {index}{source_info}]\n{chunk.text.strip()}")

        context = "\n\n".join(parts)
        logger.info("Built context from %d chunks (%d chars)", len(chunks), len(context))
        return context
