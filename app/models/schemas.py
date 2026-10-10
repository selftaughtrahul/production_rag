"""
Pydantic request / response schemas for the RAG API.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, EmailStr, Field, field_validator


# ─────────────────────────────────────────────────────────────
# Document tags (user metadata chips)
# ─────────────────────────────────────────────────────────────

MAX_TAGS = 20
MAX_TAG_LENGTH = 32
MAX_METADATA = 20
MAX_METADATA_VALUE = 64
RESERVED_TAGS = frozenset({"user_id", "document_id", "filename", "chunk_id", "chunk_index"})
_TAG_DROP = re.compile(r"[^a-z0-9_-]+")
_META_VALUE_DROP = re.compile(r"[\x00-\x1f]+")


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


def normalize_metadata(raw: Any) -> dict[str, str]:
    """Turn a JSON object or ``key: value`` pairs into short custom metadata."""
    if raw is None:
        return {}
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        if text.startswith("{"):
            try:
                raw = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError("metadata must be a JSON object") from exc
        else:
            parsed: dict[str, str] = {}
            for part in text.split(","):
                piece = part.strip()
                if not piece:
                    continue
                sep = ":" if ":" in piece else "=" if "=" in piece else ""
                if not sep:
                    continue
                key, value = piece.split(sep, 1)
                parsed[key] = value
            raw = parsed
    if not isinstance(raw, dict):
        raise ValueError("metadata must be an object")

    metadata: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        norm_key = normalize_tags([key])
        clean_key = norm_key[0] if norm_key else ""
        clean_value = _META_VALUE_DROP.sub(" ", value).strip()[:MAX_METADATA_VALUE]
        if not clean_key or not clean_value:
            continue
        metadata[clean_key] = clean_value
        if len(metadata) >= MAX_METADATA:
            break
    return metadata


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
    # tags: list[str] = Field(default_factory=list)

    # @field_validator("tags", mode="before")
    # @classmethod
    # def clean_tags(cls, v: Any) -> list[str]:
    #     return normalize_tags(v)


# ─────────────────────────────────────────────────────────────
# Document schema
# ─────────────────────────────────────────────────────────────

from datetime import datetime

DocumentStatus = Literal["processing", "ready", "failed"]


class DocumentResponse(BaseModel):
    id: str
    user_id: str
    file_name: str
    created_at: datetime
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
    file_size: int | None = None
    chunk_count: int = 0
    status: DocumentStatus = "ready"
    error_message: str | None = None


class DocumentDetailResponse(DocumentResponse):
    preview: str | None = None


class DocumentUpdate(BaseModel):
    tags: list[str] | None = None
    metadata: dict[str, str] | None = None

    @field_validator("tags", mode="before")
    @classmethod
    def clean_tags(cls, v: Any) -> list[str] | None:
        if v is None:
            return None
        return normalize_tags(v)

    @field_validator("metadata", mode="before")
    @classmethod
    def clean_metadata(cls, v: Any) -> dict[str, str] | None:
        if v is None:
            return None
        return normalize_metadata(v)


# ─────────────────────────────────────────────────────────────
# Multi-Agent schemas
# ─────────────────────────────────────────────────────────────

class GpuStatus(BaseModel):
    available: bool
    device: str
    name: str | None = None
    utilization_percent: int | None = None
    memory_used_mb: int | None = None
    memory_total_mb: int | None = None
    temperature_c: int | None = None


class QueueItem(BaseModel):
    id: str
    file_name: str
    status: DocumentStatus


class QueueStatus(BaseModel):
    broker_ok: bool
    waiting: int | None = None
    workers: int | None = None
    active: int | None = None
    processing: int = 0
    failed: int = 0
    items: list[QueueItem] = Field(default_factory=list)


class MonitorStatus(BaseModel):
    gpu: GpuStatus
    queue: QueueStatus


class ChatResponse(BaseModel):
    success: bool = True
    session_id: str
    question: str
    answer: str
    mode: Literal["agent"] = "agent"
    iterations: int = 0
    agent_trajectory: list[dict[str, Any]] = []
    guardrail_metadata: dict[str, Any] = {}
