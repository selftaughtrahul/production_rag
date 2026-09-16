"""Postgres engine, ORM session, and schema bootstrap (app state / RDS)."""

from __future__ import annotations

import logging
from collections.abc import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from database.models import Base, Document, MemoryRecord, User

logger = logging.getLogger(__name__)

_DEFAULT_DATABASE_URL = "postgresql://rag:rag@localhost:5432/rag"


def sqlalchemy_database_url(raw: str) -> str:
    """SQLAlchemy 2 + psycopg3 DSN from a standard postgresql:// URL."""
    url = raw.strip()
    if url.startswith("postgresql+psycopg://"):
        return url
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url.removeprefix("postgresql://")
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url.removeprefix("postgres://")
    raise ValueError("DATABASE_URL must be a postgresql connection string")


def checkpoint_conninfo(raw: str) -> str:
    """LangGraph Postgres checkpointer DSN (libpq postgresql://)."""
    url = raw.strip()
    if url.startswith("postgresql+psycopg://"):
        return "postgresql://" + url.removeprefix("postgresql+psycopg://")
    if url.startswith("postgres://"):
        return "postgresql://" + url.removeprefix("postgres://")
    if url.startswith("postgresql://"):
        return url
    raise ValueError("DATABASE_URL must be a postgresql connection string")


DATABASE_URL = Settings.from_environment().database_url or _DEFAULT_DATABASE_URL
CHECKPOINT_CONNINFO = checkpoint_conninfo(DATABASE_URL)
engine = create_engine(
    sqlalchemy_database_url(DATABASE_URL),
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a SQLAlchemy session."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _ensure_column(table: str, column: str, definition: str) -> None:
    """Add a column when CREATE TABLE IF NOT EXISTS is a no-op on an old schema."""
    columns = {col["name"] for col in inspect(engine).get_columns(table)}
    if column in columns:
        return
    with engine.begin() as connection:
        connection.execute(
            text(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        )


def init_db() -> None:
    """Create application tables if they do not exist."""
    Base.metadata.create_all(
        bind=engine,
        tables=[User.__table__, Document.__table__, MemoryRecord.__table__],
    )
    _ensure_column("documents", "file_path", "TEXT")
    logger.info("Postgres schema ready (users, documents, user_memories).")
