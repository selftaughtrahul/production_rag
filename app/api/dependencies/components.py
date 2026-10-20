from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from sentence_transformers import CrossEncoder

from app.core.config import Settings
from app.core.device import resolve_torch_device
from app.services.ingestion.embedding import EmbeddingService, HuggingFaceEmbeddingProvider
from app.services.vectorstore.qdrant import QdrantVectorStore


@dataclass(slots=True)
class RAGComponents:
    embedder: EmbeddingService
    vector_store: QdrantVectorStore


@lru_cache(maxsize=1)
def get_embedding_provider(model: str, device: str | None) -> HuggingFaceEmbeddingProvider:
    return HuggingFaceEmbeddingProvider(model=model, device=device)


@lru_cache(maxsize=1)
def get_cross_encoder_model(model: str, device: str | None = None) -> CrossEncoder:
    settings = Settings.from_environment()
    resolved = resolve_torch_device(device)
    token = settings.hf_token
    model_path = settings.local_model_path(model)
    local_only = Path(model_path).is_dir()
    try:
        return CrossEncoder(
            model_path,
            device=resolved,
            token=token,
            local_files_only=local_only,
        )
    except (OSError, ValueError):
        return CrossEncoder(model, device=resolved, token=token)


def warm_retrieval_models() -> None:
    """Load embedding and reranker weights so the first chat is not a cold start."""
    settings = Settings.from_environment()
    get_embedding_provider(settings.embedding_model, settings.embedding_device)
    get_cross_encoder_model(
        settings.reranker_model,
        resolve_torch_device(settings.embedding_device),
    )


@lru_cache(maxsize=1)
def build_components() -> RAGComponents:
    settings = Settings.from_environment()
    provider = get_embedding_provider(settings.embedding_model, settings.embedding_device)
    vector_store = QdrantVectorStore(
        collection_name=settings.QDRANT_COLLECTION,
        embedding_dimension=provider.dimension,
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        sparse_model=settings.QDRANT_SPARSE_MODEL,
        sparse_cache_dir=settings.models_dir,
    )
    return RAGComponents(
        embedder=EmbeddingService(provider),
        vector_store=vector_store,
    )
