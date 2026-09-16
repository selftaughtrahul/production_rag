"""Entity co-occurrence graph over retrieved chunks (in-process, no Neo4j)."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import replace
from typing import Any

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")


def _tokens(text: str) -> set[str]:
    return {match.group(0).lower() for match in _TOKEN.finditer(text or "")}


class KnowledgeGraph:
    """Build a token graph from retrieved docs and boost chunks linked to the query."""

    def __init__(self) -> None:
        self._edges: dict[str, dict[str, int]] = defaultdict(dict)

    def from_documents(self, documents: list[Any]) -> "KnowledgeGraph":
        self._edges = defaultdict(dict)
        for doc in documents:
            terms = list(_tokens(getattr(doc, "text", "") or ""))
            for left, right in zip(terms, terms[1:]):
                if left == right:
                    continue
                self._edges[left][right] = self._edges[left].get(right, 0) + 1
                self._edges[right][left] = self._edges[right].get(left, 0) + 1
        return self

    def related_terms(self, query: str, limit: int = 12) -> set[str]:
        seeds = _tokens(query)
        related: set[str] = set(seeds)
        for seed in seeds:
            neighbors = sorted(
                self._edges.get(seed, {}).items(),
                key=lambda item: item[1],
                reverse=True,
            )
            related.update(name for name, _ in neighbors[:limit])
        return related

    def rerank(self, query: str, documents: list[Any], top_k: int | None = None) -> list[Any]:
        """Stable reorder: RRF score plus a small KnowledgeGraph overlap boost."""
        if not documents:
            return []
        self.from_documents(documents)
        related = self.related_terms(query)
        scored: list[tuple[float, int, Any]] = []
        for index, doc in enumerate(documents):
            rrf = float((doc.metadata or {}).get("rrf_score", 0.0))
            overlap = len(_tokens(getattr(doc, "text", "") or "") & related)
            kg_boost = overlap / max(len(related), 1)
            total = rrf + 0.05 * kg_boost
            meta = {
                **(doc.metadata or {}),
                "kg_boost": round(kg_boost, 6),
                "kg_score": round(total, 6),
            }
            scored.append((total, index, replace(doc, metadata=meta)))
        scored.sort(key=lambda item: (-item[0], item[1]))
        ordered = [item[2] for item in scored]
        if top_k is not None:
            return ordered[:top_k]
        return ordered
