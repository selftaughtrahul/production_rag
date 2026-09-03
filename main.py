from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.auth import router as auth_router
from app.api.query import router as query_router
from database.sqlite import init_db
from app.api.documents import router as documents_router


# ─────────────────────────────────────────────────────────────
# Lifespan — runs once on startup
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the MySQL schema on startup."""
    init_db()
    yield


# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG API",
    description="Document ingestion and retrieval API with user authentication",
    version="2.0.0",
    lifespan=lifespan,
)


app.include_router(auth_router)
app.include_router(query_router)
app.include_router(documents_router)
