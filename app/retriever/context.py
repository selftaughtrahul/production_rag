from __future__ import annotations

from app.vectorstore.models import EmbeddedChunk


class ContextBuilder:
    """
    Converts retrieved chunks into LLM-ready context.
    """

    def build(
        self,
        chunks: list[EmbeddedChunk],
    ) -> str:

        if not chunks:
            return ""

        context_parts = []

        for index, chunk in enumerate(chunks, start=1):

            context_parts.append(
                f"""
[Document {index}]

{chunk.text}

Metadata:
{chunk.metadata}
"""
            )

        return "\n".join(context_parts)
