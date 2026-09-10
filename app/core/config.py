"""Central configuration for the RAG application."""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)


def _optional_env(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    
    # ── ChromaDB ──────────────────────────────
    chroma_host: str | None
    chroma_port: int
    chroma_persist_directory: str
    collection_prefix: str

    # ── Embedding ─────────────────────────────
    embedding_model: str
    embedding_device: str | None

    # ── Chunking ──────────────────────────────
    reranker_model: str
    chunk_size: int
    chunk_overlap: int

    # ── LangSmith ─────────────────────────────
    LANGSMITH_TRACING: bool
    LANGSMITH_API_KEY: str
    LANGSMITH_PROJECT: str
    LANGSMITH_ENDPOINT: str

    # ── MySQL (auth) ──────────────────────────
    mysql_host: str
    mysql_port: int
    mysql_user: str
    mysql_password: str
    mysql_database: str

    # ── JWT ───────────────────────────────────
    jwt_secret_key: str
    jwt_algorithm: str
    jwt_expire_minutes: int

    # ── Retrieval Top-K ───────────────────────
    dense_top_k: int
    bm25_top_k: int
    fusion_top_k: int
    rerank_top_k: int
    LOG_DIR:str
    IS_ASYNC:bool
    enable_nemo: bool
    nemo_config_path: str



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
            reranker_model=os.getenv(
                "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            LANGSMITH_TRACING=bool(
                os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
            ),
            LANGSMITH_API_KEY=os.getenv("LANGSMITH_API_KEY", ""),
            LANGSMITH_PROJECT=os.getenv("LANGSMITH_PROJECT", "default"),
            LANGSMITH_ENDPOINT=os.getenv(
                "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"
            ),
            # MySQL
            mysql_host=os.getenv("MYSQL_HOST", "localhost"),
            mysql_port=int(os.getenv("MYSQL_PORT", "3306")),
            mysql_user=os.getenv("MYSQL_USER", "root"),
            mysql_password=os.getenv("MYSQL_PASSWORD", ""),
            mysql_database=os.getenv("MYSQL_DATABASE", "rag_db"),
            # JWT
            jwt_secret_key=os.getenv(
                "JWT_SECRET_KEY", "change-this-secret-key-in-production"
            ),
            jwt_algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
            jwt_expire_minutes=int(os.getenv("JWT_EXPIRE_MINUTES", "60")),
            # Top-K retrieval settings
            dense_top_k=int(os.getenv("DENSE_TOP_K", "20")),
            bm25_top_k=int(os.getenv("BM25_TOP_K", "20")),
            fusion_top_k=int(os.getenv("FUSION_TOP_K", "20")),
            rerank_top_k=int(os.getenv("RERANK_TOP_K", "5")),
            IS_ASYNC=_env_flag("IS_ASYNC", "false"),
            LOG_DIR=os.getenv("LOG_DIR", "logs"),
            enable_nemo=_env_flag("ENABLE_NEMO", "true"),
            nemo_config_path=os.getenv(
                "NEMO_CONFIG_PATH",
                str(Path(__file__).resolve().parent.parent / "guardrails" / "nemo_config"),
            ),



        )

    def collection_name_for(self, embedding_dimension: int) -> str:
        model_slug = re.sub(r"[^a-z0-9]+", "_", self.embedding_model.lower()).strip("_")
        return f"{self.collection_prefix}_{model_slug}_{embedding_dimension}"

