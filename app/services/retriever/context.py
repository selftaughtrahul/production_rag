from __future__ import annotations

from app.services.vectorstore.models import EmbeddedChunk


class ContextBuilder:
    """
    Converts retrieved chunks into LLM-ready context.
    """

    def build(self,chunks: list[EmbeddedChunk],) -> str:

        if not chunks:
            return ""

        context_parts = []

        for index, chunk in enumerate(chunks, start=1):
            source = chunk.metadata.get("filename") or chunk.metadata.get("source", "")
            source_info = f" (Source: {source})" if source else ""
            context_parts.append(f"[Document {index}{source_info}]\n{chunk.text.strip()}")

        return "\n\n".join(context_parts)
