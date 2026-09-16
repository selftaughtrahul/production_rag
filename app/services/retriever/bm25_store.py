from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from sqlalchemy import String, Text, create_engine, delete, func, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from app.services.vectorstore.models import SearchResult


class Bm25Base(DeclarativeBase):
    pass


class Chunk(Bm25Base):
    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(String, primary_key=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    user_id: Mapped[str | None] = mapped_column(String, nullable=True)
    metadata_json: Mapped[str | None] = mapped_column("metadata", Text)


class BM25Store:
    """
    Persistent BM25 search using SQLite FTS5.

    Chunk rows are ORM-mapped. FTS5 MATCH has no ORM equivalent, so those
    statements use bound SQLAlchemy `text()` parameters.
    """

    def __init__(self, db_path: str = "data/bm25.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            connect_args={"check_same_thread": False},
        )
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)
        self._create_tables()

    def _create_tables(self) -> None:
        Bm25Base.metadata.create_all(bind=self.engine, tables=[Chunk.__table__])
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                    USING fts5(
                        chunk_id UNINDEXED,
                        text,
                        content='chunks',
                        content_rowid='rowid'
                    )
                    """
                )
            )

    def upsert(self, chunks: list[Any]) -> None:
        if not chunks:
            return

        session = self.SessionLocal()
        try:
            for chunk in chunks:
                metadata = getattr(chunk, "metadata", {}) or {}
                user_id = metadata.get("user_id")
                text_value = getattr(chunk, "text", "")
                chunk_id = getattr(chunk, "chunk_id", None)
                if not chunk_id or not text_value:
                    continue

                existing = session.get(Chunk, chunk_id)
                if existing is not None:
                    session.execute(
                        text(
                            """
                            DELETE FROM chunks_fts
                            WHERE rowid IN (
                                SELECT rowid FROM chunks WHERE chunk_id = :chunk_id
                            )
                            """
                        ),
                        {"chunk_id": chunk_id},
                    )
                    existing.text = text_value
                    existing.user_id = str(user_id) if user_id is not None else None
                    existing.metadata_json = json.dumps(metadata, ensure_ascii=False)
                else:
                    session.add(
                        Chunk(
                            chunk_id=chunk_id,
                            text=text_value,
                            user_id=str(user_id) if user_id is not None else None,
                            metadata_json=json.dumps(metadata, ensure_ascii=False),
                        )
                    )
                session.flush()
                rowid = session.execute(
                    text("SELECT rowid FROM chunks WHERE chunk_id = :chunk_id"),
                    {"chunk_id": chunk_id},
                ).scalar()
                if rowid is not None:
                    session.execute(
                        text(
                            """
                            INSERT INTO chunks_fts(rowid, chunk_id, text)
                            VALUES (:rowid, :chunk_id, :text)
                            """
                        ),
                        {"rowid": rowid, "chunk_id": chunk_id, "text": text_value},
                    )
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if not query.strip():
            return []
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        fts_query = self._prepare_query(query)
        if not fts_query:
            return []

        sql = """
            SELECT
                c.chunk_id,
                c.text,
                c.metadata,
                bm25(chunks_fts) AS bm25_score
            FROM chunks_fts
            JOIN chunks c ON c.rowid = chunks_fts.rowid
            WHERE chunks_fts MATCH :fts_query
        """
        params: dict[str, Any] = {"fts_query": fts_query, "top_k": top_k}
        extra_clauses: list[str] = []

        if metadata_filter:
            for index, (key, value) in enumerate(metadata_filter.items()):
                if key == "user_id":
                    extra_clauses.append("AND c.user_id = :user_id")
                    params["user_id"] = str(value)
                else:
                    extra_clauses.append(
                        f"AND json_extract(c.metadata, :json_path_{index}) = :json_value_{index}"
                    )
                    params[f"json_path_{index}"] = f"$.{key}"
                    params[f"json_value_{index}"] = value

        sql += "\n".join(extra_clauses)
        sql += "\nORDER BY bm25_score ASC\nLIMIT :top_k"

        session = self.SessionLocal()
        try:
            rows = session.execute(text(sql), params).mappings().all()
        finally:
            session.close()

        results: list[SearchResult] = []
        for row in rows:
            metadata: dict[str, Any] = {}
            if row["metadata"]:
                try:
                    metadata = json.loads(row["metadata"])
                except json.JSONDecodeError:
                    metadata = {}
            raw_score = float(row["bm25_score"])
            lexical_score = 1.0 / (1.0 + abs(raw_score))
            metadata = dict(metadata)
            metadata["bm25_score"] = raw_score
            metadata["lexical_score"] = lexical_score
            results.append(
                SearchResult(
                    chunk_id=row["chunk_id"],
                    text=row["text"],
                    score=lexical_score,
                    metadata=metadata,
                )
            )
        return results

    def delete(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return

        session = self.SessionLocal()
        try:
            for chunk_id in chunk_ids:
                session.execute(
                    text(
                        """
                        DELETE FROM chunks_fts
                        WHERE rowid IN (
                            SELECT rowid FROM chunks WHERE chunk_id = :chunk_id
                        )
                        """
                    ),
                    {"chunk_id": chunk_id},
                )
                session.execute(delete(Chunk).where(Chunk.chunk_id == chunk_id))
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def delete_document(self, document_id: str, user_id: str | None = None) -> int:
        session = self.SessionLocal()
        try:
            stmt = select(Chunk.chunk_id).where(
                func.json_extract(Chunk.metadata_json, "$.document_id") == document_id
            )
            if user_id:
                stmt = stmt.where(Chunk.user_id == user_id)
            chunk_ids = list(session.scalars(stmt).all())
        finally:
            session.close()
        self.delete(chunk_ids)
        return len(chunk_ids)

    @staticmethod
    def _prepare_query(query: str) -> str:
        query = query.strip()
        if not query:
            return ""
        tokens = re.findall(r"[A-Za-z0-9_]+", query)
        if not tokens:
            return ""
        return " OR ".join(f'"{token}"' for token in tokens)

    def close(self) -> None:
        self.engine.dispose()
