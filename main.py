"""
RAG API — application entry point.

Registers all routers and initialises the database on startup.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.auth import router as auth_router
from app.api.documents import router as documents_router
from app.api.query import router as chat_router
from database.sqlite import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the SQLite schema on startup."""
    init_db()
    yield



app = FastAPI(
    title="RAG API",
    description="Document ingestion and retrieval API with multi-agent orchestration and guardrails",
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(documents_router)

