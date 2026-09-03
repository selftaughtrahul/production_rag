from __future__ import annotations

from typing import Any

from app.services.ingestion.embedding import HuggingFaceEmbeddingProvider
from app.services.retriever.retriever import Retriever
from app.services.vectorstore.chroma import ChromaVectorStore


class RetrieverService(Retriever):
    """
    Service responsible for retrieving relevant chunks
    from the vector store.
    """

    def __init__(self,embedder: HuggingFaceEmbeddingProvider,vector_store: ChromaVectorStore,) -> None:

        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self,query: str,top_k: int = 5,metadata_filter: dict[str, Any] | None = None,) -> list[Any]:

        if not query.strip():
            return []

        # 1. Convert query into embedding
        query_embedding = self.embedder.embed_query(query)

        # 2. Search vector database
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

        return results
