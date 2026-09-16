"""
RAG API — application entry point.

Registers all routers and initialises the database on startup.
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response

from app.api.auth import router as auth_router
from app.api.documents import router as documents_router
from app.api.query import router as chat_router
from app.core.config import Settings
from app.core.logger import setup_logger
from app.core.metrics import render_prometheus
from app.core.rate_limit import attach_rate_limiter
from database.sqlite import init_db

setup_logger()
logger = logging.getLogger(__name__)


def _configure_hf_token() -> None:
    ''' Configure the Hugging Face token '''
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
    ''' Warm up the retrieval models warmup means pre-loading the models into memory '''
    from app.api.dependencies import warm_retrieval_models

    try:
        await asyncio.to_thread(warm_retrieval_models)
        logger.info("Retrieval models are ready.")
    except Exception:
        logger.exception("Background model warmup failed")


def _configure_langsmith() -> None:
    settings = Settings.from_environment()
    if not settings.LANGSMITH_TRACING or not settings.LANGSMITH_API_KEY:
        return
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY


@asynccontextmanager
async def lifespan(app: FastAPI):
    ''' Initialise the Postgres schema and warm retrieval models '''
    init_db()
    _configure_hf_token()
    _configure_langsmith()
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
attach_rate_limiter(app)


@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    """Load balancer / Compose liveness probe."""
    return {"status": "ok"}


@app.get("/metrics", tags=["ops"])
def metrics() -> Response:
    """Prometheus scrape endpoint."""
    return Response(content=render_prometheus(), media_type="text/plain; version=0.0.4")

