"""Central configuration for the RAG application."""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env", override=False)

_MIN_JWT_SECRET_LENGTH = 32
_MODEL_WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.onnx",
)


def _has_local_model_files(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    return any((path / name).is_file() for name in _MODEL_WEIGHT_FILES)


def _optional_env(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def _require_jwt_secret() -> str:
    """Return JWT_SECRET_KEY; refuse missing, blank, or short values."""
    value = os.getenv("JWT_SECRET_KEY", "").strip()
    if not value:
        raise ValueError(
            "JWT_SECRET_KEY must be set. Generate one with: "
            'python -c "import secrets; print(secrets.token_urlsafe(48))"'
        )
    if len(value) < _MIN_JWT_SECRET_LENGTH:
        raise ValueError(
            "JWT_SECRET_KEY must be at least "
            f"{_MIN_JWT_SECRET_LENGTH} characters."
        )
    return value


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
    models_dir: str
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
    database_url: str
    order_database_url: str
    rate_limit_default: str
    rate_limit_chat: str
    response_cache_ttl_seconds: int

    # ── Retrieval Top-K ───────────────────────
    dense_top_k: int
    bm25_top_k: int
    fusion_top_k: int
    rerank_top_k: int
    LOG_DIR:str
    IS_ASYNC:bool
    enable_nemo: bool
    nemo_config_path: str
    hf_token: str | None
    local_embedding: str | None
    local_reranker: str | None
    QDRANT_URL: str | None
    QDRANT_API_KEY: str | None
    QDRANT_COLLECTION: str | None


    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            chroma_host=_optional_env("CHROMA_HOST"),
            chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
            chroma_persist_directory=os.getenv(
                "CHROMA_PERSIST_DIRECTORY", "data/chroma_db"
            ),
            collection_prefix=os.getenv("RAG_COLLECTION_PREFIX", "rag"),
            models_dir=os.getenv("MODELS_DIR", r"D:\models"),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"
            ),
            embedding_device=_optional_env("EMBEDDING_DEVICE"),
            reranker_model=os.getenv(
                "RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
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
            jwt_secret_key=_require_jwt_secret(),
            jwt_algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
            jwt_expire_minutes=int(os.getenv("JWT_EXPIRE_MINUTES", "60")),
            database_url=os.getenv(
                "DATABASE_URL",
                "postgresql://rag:rag@localhost:5432/rag",
            ),
            order_database_url=os.getenv(
                "ORDER_DATABASE_URL",
                f"sqlite:///{(_PROJECT_ROOT / 'data' / 'orders_demo.db').as_posix()}",
            ),
            rate_limit_default=os.getenv("RATE_LIMIT_DEFAULT", "60/minute"),
            rate_limit_chat=os.getenv("RATE_LIMIT_CHAT", "10/minute"),
            response_cache_ttl_seconds=int(os.getenv("RESPONSE_CACHE_TTL_SECONDS", "3600")),
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
            hf_token=_optional_env("HF_TOKEN") or _optional_env("HUGGING_FACE_HUB_TOKEN"),
            local_embedding=os.getenv("LOCAL_EMBEDDING"),
            local_reranker=os.getenv("LOCAL_RERANKER"),
            QDRANT_URL = os.getenv("QDRANT_URL"),
            QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
            QDRANT_COLLECTION = os.getenv(
                "QDRANT_COLLECTION",
            ),



        )

    def local_model_path(self, model: str) -> str:
        """Prefer a pre-downloaded folder under MODELS_DIR when it exists."""
        given = Path(model)
        if _has_local_model_files(given):
            return str(given.resolve())
        candidate = Path(self.models_dir) / given.name
        if _has_local_model_files(candidate):
            return str(candidate.resolve())
        return model

    def collection_name_for(self, embedding_dimension: int) -> str:
        model_slug = re.sub(r"[^a-z0-9]+", "_", self.embedding_model.lower()).strip("_")
        return f"{self.collection_prefix}_{model_slug}_{embedding_dimension}"

