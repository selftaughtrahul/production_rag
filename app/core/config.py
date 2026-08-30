"""Central configuration for the RAG application."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Docker supplies environment variables directly. For local uvicorn/Celery runs,
# load the project .env file without replacing values already supplied by Docker.
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)


def _optional_env(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


@dataclass(frozen=True, slots=True)
class Settings:
    chroma_host: str | None
    chroma_port: int
    chroma_persist_directory: str
    collection_prefix: str
    embedding_model: str
    embedding_device: str | None
    chunk_size: int
    chunk_overlap: int
    LANGSMITH_TRACING: bool
    LANGSMITH_API_KEY: str
    LANGSMITH_PROJECT: str
    LANGSMITH_ENDPOINT: str

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            chroma_host=_optional_env("CHROMA_HOST"),
            chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
            chroma_persist_directory=os.getenv(
                "CHROMA_PERSIST_DIRECTORY", "data/chroma_db"
            ),
            collection_prefix=os.getenv("RAG_COLLECTION_PREFIX", "rag"),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            embedding_device=_optional_env("EMBEDDING_DEVICE"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            LANGSMITH_TRACING=os.getenv("LANGSMITH_TRACING"),
            LANGSMITH_API_KEY=os.getenv("LANGSMITH_API_KEY"),
            LANGSMITH_PROJECT=os.getenv("LANGSMITH_PROJECT"),
            LANGSMITH_ENDPOINT=os.getenv("LANGSMITH_ENDPOINT"),
        )

    def collection_name_for(self, embedding_dimension: int) -> str:
        model_slug = re.sub(r"[^a-z0-9]+", "_", self.embedding_model.lower()).strip("_")
        return f"{self.collection_prefix}_{model_slug}_{embedding_dimension}"
