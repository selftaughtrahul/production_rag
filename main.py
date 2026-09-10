"""
RAG API — application entry point.

Registers all routers and initialises the database on startup.
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.auth import router as auth_router
from app.api.documents import router as documents_router
from app.api.query import router as chat_router
from app.core.config import Settings
from database.sqlite import init_db

logger = logging.getLogger(__name__)


def _configure_hf_token() -> None:
    token = Settings.from_environment().hf_token
    if not token:
        logger.info(
            "HF_TOKEN is not set. Hugging Face will warn on unauthenticated downloads. "
            "Add HF_TOKEN to .env to silence that warning."
        )
        return
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token


async def _warm_retrieval_models() -> None:
    from app.api.dependencies import warm_retrieval_models

    try:
        await asyncio.to_thread(warm_retrieval_models)
        logger.info("Retrieval models are ready.")
    except Exception:
        logger.exception("Background model warmup failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the SQLite schema and warm retrieval models."""
    init_db()
    _configure_hf_token()
    warmup = asyncio.create_task(_warm_retrieval_models())
    yield
    warmup.cancel()



app = FastAPI(
    title="RAG API",
    description="Document ingestion and retrieval API with multi-agent orchestration and guardrails",
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(documents_router)

