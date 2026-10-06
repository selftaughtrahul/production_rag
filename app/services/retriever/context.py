"""
Builds the context string passed to the LLM from retrieved chunks.

Extractive only: drop duplicate / overlapping chunks, fit a character
budget, and number each kept chunk so the model can cite it as [n].
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from langsmith import traceable

from app.services.vectorstore.models import SearchResult

logger = logging.getLogger(__name__)

_WS = re.compile(r"\s+")


@dataclass(slots=True, frozen=True)
class SourceRef:
    """Citation target for one numbered context block. No chunk text on the wire."""

    index: int
    filename: str
    page: int | None
    chunk_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "filename": self.filename,
            "page": self.page,
            "chunk_id": self.chunk_id,
        }


@dataclass(slots=True, frozen=True)
class BuiltContext:
    text: str
    sources: list[SourceRef]


def _normalize(text: str) -> str:
    return _WS.sub(" ", (text or "").strip()).lower()


def _page_of(metadata: dict[str, Any]) -> int | None:
    for key in ("page", "page_number"):
        value = metadata.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    return None


def _filename_of(metadata: dict[str, Any]) -> str:
    return str(metadata.get("filename") or metadata.get("source") or "unknown")


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class ContextBuilder:
    """
    Converts retrieved chunks into a single numbered context string plus
    the list of sources those numbers point to.
    """

    def __init__(self, max_chars: int | None = None, near_duplicate_threshold: float = 0.9) -> None:
        self.max_chars = max_chars
        self.near_duplicate_threshold = near_duplicate_threshold

    def _budget(self) -> int:
        if self.max_chars is not None:
            return self.max_chars
        # Lazy import: Settings requires JWT_SECRET_KEY; tests may construct the builder directly.
        from app.core.config import Settings

        return Settings.from_environment().context_max_chars

    def dedupe(self, chunks: list[SearchResult]) -> list[SearchResult]:
        """Keep the higher-scored chunk; drop exact, contained, and near-duplicate text."""
        ordered = sorted(chunks, key=lambda c: c.score, reverse=True)
        kept: list[SearchResult] = []
        seen_ids: set[str] = set()
        kept_norm: list[str] = []
        kept_words: list[set[str]] = []

        for chunk in ordered:
            norm = _normalize(chunk.text)
            if not norm:
                continue
            if chunk.chunk_id and chunk.chunk_id in seen_ids:
                continue
            words = set(norm.split())
            duplicate = False
            superseded: int | None = None
            for pos, (other_norm, other_words) in enumerate(zip(kept_norm, kept_words)):
                if norm == other_norm or norm in other_norm:
                    duplicate = True
                    break
                if other_norm in norm:
                    # The new chunk is a superset of a kept one: keep the longer text.
                    superseded = pos
                    break
                if _jaccard(words, other_words) >= self.near_duplicate_threshold:
                    duplicate = True
                    break
            if duplicate:
                continue
            if superseded is not None:
                del kept[superseded], kept_norm[superseded], kept_words[superseded]
            kept.append(chunk)
            if chunk.chunk_id:
                seen_ids.add(chunk.chunk_id)
            kept_norm.append(norm)
            kept_words.append(words)

        # Restore the reranker's order among the survivors.
        survivors = {id(c) for c in kept}
        return [c for c in chunks if id(c) in survivors]

    @traceable(run_type="chain", name="Build Context String")
    def build(self, chunks: list[SearchResult]) -> BuiltContext:
        if not chunks:
            return BuiltContext(text="", sources=[])

        unique = self.dedupe(chunks)
        budget = self._budget()

        parts: list[str] = []
        sources: list[SourceRef] = []
        used = 0
        for chunk in unique:
            index = len(parts) + 1
            filename = _filename_of(chunk.metadata)
            page = _page_of(chunk.metadata)
            label = f"[{index}] {filename}" + (f", page {page}" if page is not None else "")
            block = f"{label}\n{chunk.text.strip()}"
            cost = len(block) + (2 if parts else 0)
            if parts and used + cost > budget:
                break
            parts.append(block)
            used += cost
            sources.append(SourceRef(index=index, filename=filename, page=page, chunk_id=chunk.chunk_id))

        context = "\n\n".join(parts)
        logger.info(
            "Built context: %d chunks in, %d after dedupe, %d kept within %d chars (%d chars)",
            len(chunks), len(unique), len(parts), budget, len(context),
        )
        return BuiltContext(text=context, sources=sources)
