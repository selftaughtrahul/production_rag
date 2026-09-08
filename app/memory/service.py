from __future__ import annotations

import sqlite3
from uuid import uuid4

from app.memory.models import UserMemory


class MemoryService:
    """
    User Memory Service
    manages CRUD and retrieval of user memories
    """
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def get_user_memories(self, user_id: str, limit: int = 20) -> list[UserMemory]:
        ''' Retrieve active memories for a user, ordered by importance '''
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT id, user_id, memory, memory_type, importance
            FROM user_memories
            WHERE user_id = ? AND is_active = 1
            ORDER BY importance DESC, updated_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        return [UserMemory(**dict(row)) for row in cursor.fetchall()]

    def create_memory(self, user_id: str, memory: str, memory_type: str, importance: float) -> UserMemory:
        ''' Create memories for a user '''

        memory_id = str(uuid4())
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO user_memories (
                id, user_id, memory, memory_type, importance, is_active
            )
            VALUES (?, ?, ?, ?, ?, 1)
            """,
            (memory_id, user_id, memory, memory_type, importance),
        )
        self.conn.commit()
        return UserMemory(
            id=memory_id,
            user_id=user_id,
            memory=memory,
            memory_type=memory_type,
            importance=importance,
        )

    def update_memory(self, memory_id: str, memory: str, memory_type: str, importance: float,user_id: str | None = None) -> UserMemory | None:
        ''' Update memories for a user '''

        cursor = self.conn.cursor()
        if user_id:
            cursor.execute(
                """
                UPDATE user_memories
                SET memory = ?,
                    memory_type = ?,
                    importance = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND user_id = ? AND is_active = 1
                """,
                (memory, memory_type, importance, memory_id, user_id),
            )
        else:
            cursor.execute(
                """
                UPDATE user_memories
                SET memory = ?,
                    memory_type = ?,
                    importance = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND is_active = 1
                """,
                (memory, memory_type, importance, memory_id),
            )
        self.conn.commit()
        if cursor.rowcount == 0:
            return None

        cursor.execute(
            """
            SELECT id, user_id, memory, memory_type, importance
            FROM user_memories
            WHERE id = ?
            """,
            (memory_id,),
        )
        row = cursor.fetchone()
        return UserMemory(**dict(row)) if row else None

    def delete_memory(self, user_id: str, memory_id: str) -> bool:
        ''' Delete memories for a user '''

        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE user_memories
            SET is_active = 0,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND user_id = ? AND is_active = 1
            """,
            (memory_id, user_id),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def clear_user_memories(self, user_id: str) -> None:
        ''' Delete all memories for a user '''

        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE user_memories
            SET is_active = 0,
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id = ? AND is_active = 1
            """,
            (user_id,),
        )
        self.conn.commit()
