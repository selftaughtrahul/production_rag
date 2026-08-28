from __future__ import annotations
from app.vectorstore.chroma import ChromaVectorStore


class IngestionPipeline:
    """
    Orchestrates the complete document ingestion pipeline.
    """

    def __init__(
        self,
        loader,
        cleaner,
        chunker,
        embedder,
        vector_store: ChromaVectorStore,
    ) -> None:

        self.loader = loader
        self.cleaner = cleaner
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

    async def ingest(
        self,
        source: str,
    ) -> dict:

        # 1. Load document
        document = self.loader.load(source)

        # 2. Clean document text
        cleaned_text = self.cleaner.clean(document.text)

        # 3. Split into chunks with document metadata
        chunks = self.chunker.chunk(cleaned_text, metadata=document.metadata)

        # 4. Generate embeddings
        embedded_chunks = self.embedder.embed_chunks(chunks)

        # 5. Store embeddings
        self.vector_store.add(embedded_chunks)

        return {
            "chunks_count": len(chunks),
            "document": document.metadata,
        }
