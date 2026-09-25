"""SQLite engine, ORM session, and schema bootstrap (users, documents, memories).

The LangGraph checkpointer writes to the same SQLite file, so session
ownership checks can read its ``checkpoints`` table through SessionLocal.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from database.models import Base, Document, MemoryRecord, User

logger = logging.getLogger(__name__)


def sqlite_file_path(database_url: str) -> str:
    """Return the file path of a sqlite:/// URL; refuse anything else."""
    url = make_url(database_url)
    if url.get_backend_name() != "sqlite" or not url.database:
        raise ValueError("DATABASE_URL must be a file-based sqlite:/// URL")
    return url.database


DATABASE_URL = Settings.from_environment().database_url
CHECKPOINT_DB_PATH = sqlite_file_path(DATABASE_URL)
Path(CHECKPOINT_DB_PATH).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30},
)


@event.listens_for(engine, "connect")
def _configure_sqlite(dbapi_connection: Any, _record: Any) -> None:
    # WAL lets the API, Celery worker, and checkpointer read while one writes.
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.close()


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
    logger.info("SQLite schema ready at %s (users, documents, user_memories).", CHECKPOINT_DB_PATH)
