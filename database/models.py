"""SQLAlchemy ORM models for application state (SQLite)."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    username: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    password: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.current_timestamp()
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    documents: Mapped[list["Document"]] = relationship(back_populates="user")
    memories: Mapped[list["MemoryRecord"]] = relationship(back_populates="user")


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        Index("idx_documents_user_hash", "user_id", "content_hash", unique=True),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    file_name: Mapped[str] = mapped_column(String, nullable=False)
    file_path: Mapped[str | None] = mapped_column(String, nullable=True)
    # JSON-encoded list of user tags; mirrored onto every chunk as metadata.tags
    tags: Mapped[str | None] = mapped_column(Text, nullable=True)
    # JSON object of user key/value chips, shown on the document
    custom_metadata: Mapped[str | None] = mapped_column(Text, nullable=True)
    # processing | ready | failed. Existing rows are already ingested.
    status: Mapped[str] = mapped_column(
        String, nullable=False, default="ready", server_default="ready"
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    # SHA-256 of the upload bytes; unique per user so the same file is rejected
    content_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.current_timestamp()
    )

    user: Mapped[User] = relationship(back_populates="documents")


class MemoryRecord(Base):
    __tablename__ = "user_memories"
    __table_args__ = (
        Index("idx_user_active_memories", "user_id", "is_active"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    memory: Mapped[str] = mapped_column(Text, nullable=False)
    memory_type: Mapped[str] = mapped_column(String, nullable=False, default="general")
    importance: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.current_timestamp()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
    )

    user: Mapped[User] = relationship(back_populates="memories")
