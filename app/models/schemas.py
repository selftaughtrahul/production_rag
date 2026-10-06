"""
Pydantic request / response schemas for the RAG API.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, EmailStr, Field, field_validator


# ─────────────────────────────────────────────────────────────
# Document tags (user metadata chips)
# ─────────────────────────────────────────────────────────────

MAX_TAGS = 20
MAX_TAG_LENGTH = 32
RESERVED_TAGS = frozenset({"user_id", "document_id", "filename", "chunk_id", "chunk_index"})
_TAG_DROP = re.compile(r"[^a-z0-9_-]+")


def normalize_tags(raw: Any) -> list[str]:
    """Lowercase, hyphenate spaces, strip unsafe chars, dedupe, cap count and length."""
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple, set)):
        raise ValueError("tags must be a list of strings")

    tags: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        tag = re.sub(r"\s+", "-", item.strip().lower())
        tag = _TAG_DROP.sub("", tag).strip("-_")[:MAX_TAG_LENGTH]
        if not tag or tag in RESERVED_TAGS or tag in tags:
            continue
        tags.append(tag)
        if len(tags) >= MAX_TAGS:
            break
    return tags


# ─────────────────────────────────────────────────────────────
# Auth schemas
# ─────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def password_min_length(cls, v: str) -> str:
        if len(v) < 5:
            raise ValueError("Password must be at least 5 characters")
        return v

    @field_validator("username")
    @classmethod
    def username_min_length(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("Username must be at least 3 characters")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    username: str


class UserInDB(BaseModel):
    """Representation of a user row returned from MySQL."""
    id: str
    username: str
    email: str
    is_active: bool


MemoryType = Literal[
    "personal",
    "professional",
    "technical",
    "preference",
    "project",
    "goal",
    "general",
    "semantic",
    "episodic",
]


class MemoryItem(BaseModel):
    id: str
    memory: str
    memory_type: str
    importance: float


class MemoryCreate(BaseModel):
    memory: str
    memory_type: MemoryType = "general"

    @field_validator("memory")
    @classmethod
    def memory_text(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("Memory cannot be empty")
        if len(text) > 500:
            raise ValueError("Memory must be 500 characters or fewer")
        return text


class MemoryUpdate(BaseModel):
    memory: str
    memory_type: MemoryType

    @field_validator("memory")
    @classmethod
    def memory_text(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("Memory cannot be empty")
        if len(text) > 500:
            raise ValueError("Memory must be 500 characters or fewer")
        return text


# ─────────────────────────────────────────────────────────────
# Query schema
# ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None
    tags: list[str] = Field(default_factory=list)

    @field_validator("tags", mode="before")
    @classmethod
    def clean_tags(cls, v: Any) -> list[str]:
        return normalize_tags(v)


# ─────────────────────────────────────────────────────────────
# Document schema
# ─────────────────────────────────────────────────────────────

from datetime import datetime

class DocumentResponse(BaseModel):
    id: str
    user_id: str
    file_name: str
    created_at: datetime
    tags: list[str] = Field(default_factory=list)
    file_size: int | None = None
    chunk_count: int = 0


# ─────────────────────────────────────────────────────────────
# Multi-Agent schemas
# ─────────────────────────────────────────────────────────────

class ChatResponse(BaseModel):
    success: bool = True
    session_id: str
    question: str
    answer: str
    mode: Literal["agent"] = "agent"
    iterations: int = 0
    agent_trajectory: list[dict[str, Any]] = []
    guardrail_metadata: dict[str, Any] = {}
