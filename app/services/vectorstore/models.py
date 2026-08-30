from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class EmbeddedChunk:
    """
    A text chunk together with its vector embedding.
    """

    chunk_id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class SearchResult:
    """
    Standardized result returned by every vector store.
    """

    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)