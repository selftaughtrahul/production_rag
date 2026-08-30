from __future__ import annotations
from app.services.vectorstore.chroma import ChromaVectorStore
from typing import Any


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

    def ingest(self, source: str, document_id: str, filename: str) -> dict[str, Any]:
        # -----------------------------------------------------
        # 1. Load document
        # -----------------------------------------------------
        document = self.loader.load(source)

        # -----------------------------------------------------
        # 2. Add document-level metadata
        # -----------------------------------------------------
        document_metadata = {
            **document.metadata,
            "document_id": document_id,
            "filename": filename,
        }

        # -----------------------------------------------------
        # 3. Clean document
        # -----------------------------------------------------
        cleaned_text = self.cleaner.clean(document.text)

        # -----------------------------------------------------
        # 4. Create chunks
        # -----------------------------------------------------
        chunks = self.chunker.chunk(cleaned_text, metadata=document_metadata)

        # -----------------------------------------------------
        # 5. Make sure every chunk has document_id
        # -----------------------------------------------------
        for index, chunk in enumerate(chunks):
            chunk.metadata["document_id"] = document_id
            chunk.metadata["filename"] = filename
            chunk.metadata["chunk_index"] = index
            chunk.metadata["chunk_id"] = f"{document_id}_chunk_{index}"

        # -----------------------------------------------------
        # 6. Generate embeddings
        # -----------------------------------------------------
        embedded_chunks = self.embedder.embed_chunks(chunks)

        # -----------------------------------------------------
        # 7. Store vectors
        # -----------------------------------------------------
        self.vector_store.add(embedded_chunks)

        return {
            "document_id": document_id,
            "filename": filename,
            "chunks_count": len(chunks),
        }
