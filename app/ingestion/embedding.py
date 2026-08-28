from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from .chunker import Chunk
from langchain_openai import OpenAIEmbeddings


class EmbeddingProvider(ABC):
    """
    Abstract interface for all embedding providers.

    Every provider must implement:
        - embed_documents()
        - embed_query()
    """

    @abstractmethod
    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple documents.
        """
        raise NotImplementedError

    @abstractmethod
    def embed_query(
        self,
        query: str,
    ) -> list[float]:
        """
        Generate an embedding for a query.
        """
        raise NotImplementedError


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI embedding provider.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
    ):
        self.model = model

        self.embeddings = OpenAIEmbeddings(
            model=model,
        )

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        return self.embeddings.embed_documents(texts)

    def embed_query(
        self,
        query: str,
    ) -> list[float]:

        return self.embeddings.embed_query(query)


class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    """
    Local Sentence Transformers / Hugging Face embedding provider.
    Runs locally on CPU/GPU without requiring an API key.

    Example models:
        sentence-transformers/all-MiniLM-L6-v2 (dim: 384)
        BAAI/bge-small-en-v1.5 (dim: 384)
        BAAI/bge-base-en-v1.5 (dim: 768)
    """

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str | None = None,
    ):
        import torch
        from sentence_transformers import SentenceTransformer

        # Auto-detect CUDA if available; otherwise safely use CPU
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        self.model_name = model
        self.device = device
        self.model = SentenceTransformer(model, device=device)

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(
        self,
        query: str,
    ) -> list[float]:
        embedding = self.model.encode([query], convert_to_numpy=True)[0]
        return embedding.tolist()


class OllamaEmbeddingProvider(EmbeddingProvider):
    """
    Local Ollama embedding provider.

    Example models:
        nomic-embed-text
        mxbai-embed-large
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        *,
        base_url: str = "http://localhost:11434",
    ):
        self.model = model

        self.embeddings = OllamaEmbeddings(
            model=model,
            base_url=base_url,
        )

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        return self.embeddings.embed_documents(texts)

    def embed_query(
        self,
        query: str,
    ) -> list[float]:

        return self.embeddings.embed_query(query)


@dataclass(slots=True)
class EmbeddedChunk:
    """
    Standard representation of an embedded chunk.
    """

    text: str
    embedding: list[float]
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        """
        Return unique chunk ID based on metadata and index.
        """
        filename = self.metadata.get("filename", "doc")
        return f"{filename}_{self.index}"

    @property
    def dimension(self) -> int:
        """
        Return embedding vector dimension.
        """

        return len(self.embedding)


class EmbeddingService:
    """
    Application-level embedding service.

    The service does not know which embedding model
    is being used.

    It depends only on the EmbeddingProvider interface.
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
    ):
        self.provider = provider

    def embed_chunks(
        self,
        chunks: list[Chunk],
    ) -> list[EmbeddedChunk]:

        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]

        embeddings = self.provider.embed_documents(texts)

        if len(embeddings) != len(chunks):
            raise RuntimeError("Embedding count does not match " "chunk count")

        return [
            EmbeddedChunk(
                text=chunk.text,
                embedding=embedding,
                index=chunk.index,
                metadata=chunk.metadata.copy(),
            )
            for chunk, embedding in zip(
                chunks,
                embeddings,
                strict=True,
            )
        ]

    def embed_query(
        self,
        query: str,
    ) -> list[float]:

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if not query.strip():
            raise ValueError("query cannot be empty")

        return self.provider.embed_query(query)
