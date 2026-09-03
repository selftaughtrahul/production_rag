from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from app.services.vectorstore.models import SearchResult


class BM25Store:
    """
    Persistent BM25 search using SQLite FTS5.

    This stores:
        - chunk_id
        - text
        - metadata
        - user_id

    and provides lexical BM25 retrieval.
    """

    def __init__(
        self,
        db_path: str = "data/bm25.db",
    ) -> None:

        self.db_path = Path(db_path)

        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.connection = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
        )

        self.connection.row_factory = sqlite3.Row

        self._create_tables()



    def _create_tables(self) -> None:

        cursor = self.connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                user_id TEXT,
                metadata TEXT
            )
            """
        )

        cursor.execute(
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

        self.connection.commit()


    def upsert(
        self,
        chunks: list[Any],
    ) -> None:

        if not chunks:
            return

        cursor = self.connection.cursor()

        for chunk in chunks:

            metadata = getattr(
                chunk,
                "metadata",
                {},
            ) or {}

            user_id = metadata.get(
                "user_id"
            )

            text = getattr(
                chunk,
                "text",
                "",
            )

            chunk_id = getattr(
                chunk,
                "chunk_id",
                None,
            )

            if not chunk_id or not text:
                continue

            # Remove old FTS entry if it exists
            cursor.execute(
                """
                DELETE FROM chunks_fts
                WHERE rowid IN (
                    SELECT rowid
                    FROM chunks
                    WHERE chunk_id = ?
                )
                """,
                (chunk_id,),
            )

            # Upsert main record
            cursor.execute(
                """
                INSERT INTO chunks (
                    chunk_id,
                    text,
                    user_id,
                    metadata
                )
                VALUES (?, ?, ?, ?)

                ON CONFLICT(chunk_id)
                DO UPDATE SET
                    text = excluded.text,
                    user_id = excluded.user_id,
                    metadata = excluded.metadata
                """,
                (
                    chunk_id,
                    text,
                    str(user_id) if user_id is not None else None,
                    json.dumps(
                        metadata,
                        ensure_ascii=False,
                    ),
                ),
            )

            # Get rowid
            cursor.execute(
                """
                SELECT rowid
                FROM chunks
                WHERE chunk_id = ?
                """,
                (chunk_id,),
            )

            row = cursor.fetchone()

            if row:

                cursor.execute(
                    """
                    INSERT INTO chunks_fts(
                        rowid,
                        chunk_id,
                        text
                    )
                    VALUES (?, ?, ?)
                    """,
                    (
                        row["rowid"],
                        chunk_id,
                        text,
                    ),
                )

        self.connection.commit()



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
            raise ValueError(
                "top_k must be greater than 0"
            )

        fts_query = self._prepare_query(query)

        if not fts_query:
            return []

        sql = """
            SELECT
                c.chunk_id,
                c.text,
                c.metadata,

                bm25(
                    chunks_fts
                ) AS bm25_score

            FROM chunks_fts

            JOIN chunks c
                ON c.rowid = chunks_fts.rowid

            WHERE chunks_fts MATCH ?
        """

        params: list[Any] = [
            fts_query,
        ]

        # -----------------------------------------------------
        # USER SCOPING
        # -----------------------------------------------------

        if metadata_filter:

            for key, value in metadata_filter.items():

                if key == "user_id":

                    sql += """
                        AND c.user_id = ?
                    """

                    params.append(
                        str(value)
                    )

                else:

                    # SQLite JSON metadata filtering
                    sql += """
                        AND json_extract(
                            c.metadata,
                            ?
                        ) = ?
                    """

                    params.append(
                        f"$.{key}"
                    )

                    params.append(
                        value
                    )

        sql += """
            ORDER BY bm25_score ASC
            LIMIT ?
        """

        params.append(top_k)

        cursor = self.connection.cursor()

        cursor.execute(
            sql,
            params,
        )

        rows = cursor.fetchall()

        results: list[SearchResult] = []

        for row in rows:

            metadata = {}

            if row["metadata"]:

                try:

                    metadata = json.loads(
                        row["metadata"]
                    )

                except json.JSONDecodeError:

                    metadata = {}

            # SQLite FTS5 bm25() returns
            # lower / more negative values for
            # better matches.
            #
            # We store a positive normalized
            # lexical score for debugging.
            raw_score = float(
                row["bm25_score"]
            )

            lexical_score = 1.0 / (
                1.0 + abs(raw_score)
            )

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


    def delete(
        self,
        chunk_ids: list[str],
    ) -> None:

        if not chunk_ids:
            return

        cursor = self.connection.cursor()

        for chunk_id in chunk_ids:

            cursor.execute(
                """
                DELETE FROM chunks_fts
                WHERE rowid IN (
                    SELECT rowid
                    FROM chunks
                    WHERE chunk_id = ?
                )
                """,
                (chunk_id,),
            )

            cursor.execute(
                """
                DELETE FROM chunks
                WHERE chunk_id = ?
                """,
                (chunk_id,),
            )

        self.connection.commit()


    def delete_document(
        self,
        document_id: str,
    ) -> int:

        cursor = self.connection.cursor()

        cursor.execute(
            """
            SELECT chunk_id
            FROM chunks
            WHERE json_extract(
                metadata,
                '$.document_id'
            ) = ?
            """,
            (document_id,),
        )

        rows = cursor.fetchall()

        chunk_ids = [
            row["chunk_id"]
            for row in rows
        ]

        self.delete(chunk_ids)

        return len(chunk_ids)

    @staticmethod
    def _prepare_query(
        query: str,
    ) -> str:

        query = query.strip()

        if not query:
            return ""

        # Extract words/numbers rather than passing
        # arbitrary punctuation to FTS5.
        tokens = re.findall(
            r"[A-Za-z0-9_]+",
            query,
        )

        if not tokens:
            return ""

        # OR gives better recall for natural-language
        # queries than requiring every token.
        return " OR ".join(
            f'"{token}"'
            for token in tokens
        )



    def close(self) -> None:

        self.connection.close()