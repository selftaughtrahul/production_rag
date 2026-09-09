"""
SQLite database connection module.

Provides:
    - get_connection() -> raw SQLite connection
    - get_db() -> FastAPI dependency yielding (connection, cursor)
    - init_db() -> creates application tables if they do not exist
"""

from __future__ import annotations

from typing import Generator
import sqlite3
from sqlalchemy import create_engine

from app.core.config import Settings

DB_FILE_PATH = "rag_database.db"
engine = create_engine(
    f"sqlite:///{DB_FILE_PATH}",
    connect_args={"check_same_thread": False},
)


def get_connection() -> sqlite3.Connection:
    """
    Create and return a new SQLite database connection.
    """

    conn = sqlite3.connect(
        DB_FILE_PATH,
        check_same_thread=False,
    )

    # Allows rows to be accessed like dictionaries:
    # row["username"]
    # row["email"]
    conn.row_factory = sqlite3.Row

    return conn


def get_db() -> Generator[
    tuple[sqlite3.Connection, sqlite3.Cursor],
    None,
    None,
]:
    """
    FastAPI database dependency.

    Usage:

        @router.get("/users")
        def get_users(db=Depends(get_db)):
            conn, cursor = db

            cursor.execute("SELECT * FROM users")

            rows = cursor.fetchall()

            return [dict(row) for row in rows]
    """

    conn = get_connection()
    cursor = conn.cursor()

    try:
        yield conn, cursor

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        cursor.close()
        conn.close()


def init_db() -> None:
    """
    Create application database tables if they do not exist.
    """

    conn = get_connection()
    cursor = conn.cursor()

    try:

        # Users table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
            """
        )

        # Documents table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                file_name TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (user_id)
                    REFERENCES users(id)
                    ON DELETE CASCADE
            )
            """
        )

        # User Memories table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                memory TEXT NOT NULL,
                memory_type TEXT NOT NULL DEFAULT 'general',
                importance REAL NOT NULL DEFAULT 0.5,
                is_active BOOLEAN NOT NULL DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (user_id)
                    REFERENCES users(id)
                    ON DELETE CASCADE
            )
            """
        )
        
        # Index on user_id and is_active for user_memories
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_user_active_memories 
            ON user_memories (user_id, is_active)
            """
        )

        conn.commit()

        print("[DB] users table ready.")
        print("[DB] documents table ready.")
        print("[DB] user_memories table ready.")

    except Exception:
        conn.rollback()
        raise

    finally:
        cursor.close()
        conn.close()