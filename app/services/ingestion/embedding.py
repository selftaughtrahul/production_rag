from __future__ import annotations
from abc import ABC, abstractmethod
from app.services.vectorstore.models import EmbeddedChunk
from .chunker import Chunk
from langchain_openai import OpenAIEmbeddings

class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError

class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self.embeddings = OpenAIEmbeddings(model=model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embeddings.embed_documents(texts)

    def embed_query(self, query: str) -> list[float]:
        return self.embeddings.embed_query(query)

    @property
    def dimension(self) -> int:
        return 1536

class HuggingFaceEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2", *, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        from app.core.device import resolve_torch_device

        from app.core.config import Settings

        device = resolve_torch_device(device)
        token = Settings.from_environment().hf_token
        self.model_name = model
        self.device = device
        try:
            self.model = SentenceTransformer(
                model, device=device, token=token, local_files_only=True
            )
        except Exception:
            self.model = SentenceTransformer(model, device=device, token=token)
        dimension_getter = getattr(self.model, "get_embedding_dimension", None)
        if dimension_getter is None:
            dimension_getter = self.model.get_sentence_embedding_dimension
        self._dimension = dimension_getter()
        if not self._dimension:
            raise RuntimeError(f"Could not determine embedding dimension for {model}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [] if not texts else self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.model.encode([query], convert_to_numpy=True)[0].tolist()

    @property
    def dimension(self) -> int:
        return self._dimension

class EmbeddingService:
    def __init__(self, provider: EmbeddingProvider):
        self.provider = provider

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        if not chunks:
            return []
        embeddings = self.provider.embed_documents([chunk.text for chunk in chunks])
        if len(embeddings) != len(chunks):
            raise RuntimeError("Embedding count does not match chunk count")
        return [EmbeddedChunk(
            chunk_id=self._chunk_id(chunk), text=chunk.text, embedding=embedding,
            metadata=chunk.metadata.copy(),
        ) for chunk, embedding in zip(chunks, embeddings, strict=True)]

    def embed_query(self, query: str) -> list[float]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query cannot be empty")
        return self.provider.embed_query(query)

    @staticmethod
    def _chunk_id(chunk: Chunk) -> str:
        source = str(chunk.metadata.get("source", chunk.metadata.get("filename", "document")))
        return f"{source}:{chunk.index}"
