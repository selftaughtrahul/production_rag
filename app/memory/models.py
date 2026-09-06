from typing import Literal

from pydantic import BaseModel, Field


class MemoryDecision(BaseModel):
    action: Literal["ADD", "UPDATE", "IGNORE"]
    memory: str | None = None
    memory_id: str | None = None
    memory_type: Literal[
        "personal",
        "professional",
        "technical",
        "preference",
        "project",
        "goal",
        "general",
    ] | None = None
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    reason: str = ""


class UserMemory(BaseModel):
    id: str
    user_id: str
    memory: str
    memory_type: str
    importance: float
