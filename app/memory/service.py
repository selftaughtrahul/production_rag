from __future__ import annotations

from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.orm import Session
from sqlalchemy.sql import func

from app.memory.models import UserMemory
from database.models import MemoryRecord


class MemoryService:
    """
    User Memory Service
    manages CRUD and retrieval of user memories
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_user_memories(self, user_id: str, limit: int = 20) -> list[UserMemory]:
        """Retrieve active memories for a user, ordered by importance."""
        rows = self.session.scalars(
            select(MemoryRecord)
            .where(MemoryRecord.user_id == user_id, MemoryRecord.is_active.is_(True))
            .order_by(MemoryRecord.importance.desc(), MemoryRecord.updated_at.desc())
            .limit(limit)
        ).all()
        return [
            UserMemory(
                id=row.id,
                user_id=row.user_id,
                memory=row.memory,
                memory_type=row.memory_type,
                importance=row.importance,
            )
            for row in rows
        ]

    def create_memory(
        self, user_id: str, memory: str, memory_type: str, importance: float
    ) -> UserMemory:
        """Create memories for a user."""
        memory_id = str(uuid4())
        record = MemoryRecord(
            id=memory_id,
            user_id=user_id,
            memory=memory,
            memory_type=memory_type,
            importance=importance,
            is_active=True,
        )
        self.session.add(record)
        self.session.commit()
        return UserMemory(
            id=memory_id,
            user_id=user_id,
            memory=memory,
            memory_type=memory_type,
            importance=importance,
        )

    def update_memory(
        self,
        memory_id: str,
        memory: str,
        memory_type: str,
        importance: float,
        user_id: str | None = None,
    ) -> UserMemory | None:
        """Update memories for a user."""
        filters = [
            MemoryRecord.id == memory_id,
            MemoryRecord.is_active.is_(True),
        ]
        if user_id:
            filters.append(MemoryRecord.user_id == user_id)

        result = self.session.execute(
            update(MemoryRecord)
            .where(*filters)
            .values(
                memory=memory,
                memory_type=memory_type,
                importance=importance,
                updated_at=func.current_timestamp(),
            )
        )
        self.session.commit()
        if result.rowcount == 0:
            return None

        row = self.session.get(MemoryRecord, memory_id)
        if row is None:
            return None
        return UserMemory(
            id=row.id,
            user_id=row.user_id,
            memory=row.memory,
            memory_type=row.memory_type,
            importance=row.importance,
        )

    def delete_memory(self, user_id: str, memory_id: str) -> bool:
        """Delete memories for a user."""
        result = self.session.execute(
            update(MemoryRecord)
            .where(
                MemoryRecord.id == memory_id,
                MemoryRecord.user_id == user_id,
                MemoryRecord.is_active.is_(True),
            )
            .values(is_active=False, updated_at=func.current_timestamp())
        )
        self.session.commit()
        return result.rowcount > 0

    def clear_user_memories(self, user_id: str) -> None:
        """Delete all memories for a user."""
        self.session.execute(
            update(MemoryRecord)
            .where(
                MemoryRecord.user_id == user_id,
                MemoryRecord.is_active.is_(True),
            )
            .values(is_active=False, updated_at=func.current_timestamp())
        )
        self.session.commit()
