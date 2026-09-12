"""SQLite engine, ORM session, and schema bootstrap."""

from __future__ import annotations

import logging
from collections.abc import Generator

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from database.models import Base, Document, MemoryRecord, User

logger = logging.getLogger(__name__)

DB_FILE_PATH = "rag_database.db"
engine = create_engine(
    f"sqlite:///{DB_FILE_PATH}",
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


@event.listens_for(engine, "connect")
def _enable_foreign_keys(dbapi_connection, _connection_record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


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
    """Add a column when CREATE TABLE IF NOT EXISTS is a no-op on an old file."""
    columns = {col["name"] for col in inspect(engine).get_columns(table)}
    if column in columns:
        return
    with engine.begin() as connection:
        connection.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {definition}"))


def init_db() -> None:
    """Create application tables if they do not exist."""
    Base.metadata.create_all(
        bind=engine,
        tables=[User.__table__, Document.__table__, MemoryRecord.__table__],
    )
    _ensure_column("documents", "file_path", "TEXT")
    logger.info("SQLite schema ready (users, documents, user_memories).")
