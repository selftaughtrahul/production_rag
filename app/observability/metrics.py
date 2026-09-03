from __future__ import annotations

import time

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RAGMetrics:
    """
    Runtime metrics for a single RAG request.
    """

    start_time: float = field(default_factory=time.perf_counter)
    retrieved_documents: int = 0
    relevant_documents: int = 0
    retry_count: int = 0
    query_rewritten: bool = False
    answer_generated: bool = False
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def finish(self) -> None:
        """
        Calculate total request latency.
        """

        self.total_latency_ms = (time.perf_counter() - self.start_time) * 1000

    def record_retrieval(self,document_count: int,latency_ms: float | None = None,) -> None:
        """
        Record retrieval information.
        """

        self.retrieved_documents = document_count

        if latency_ms is not None:
            self.retrieval_latency_ms = latency_ms

    def record_grading(self,relevant_count: int,) -> None:
        """
        Record document grading result.
        """

        self.relevant_documents = relevant_count

    def record_rewrite(self) -> None:
        """
        Record that query rewriting occurred.
        """

        self.query_rewritten = True

    def record_retry(self) -> None:
        """
        Increment retry count.
        """

        self.retry_count += 1

    def record_generation(self,latency_ms: float | None = None,) -> None:
        """
        Record answer generation.
        """

        self.answer_generated = True

        if latency_ms is not None:
            self.generation_latency_ms = latency_ms

    def set_extra(self,key: str,value: Any,) -> None:
        """
        Store additional application-specific metadata.
        """

        self.extra[key] = value

    def finish_and_get(self) -> dict[str, Any]:
        """
        Finish metrics and return a serializable dictionary.
        """

        self.finish()

        return {
            "retrieved_documents": self.retrieved_documents,
            "relevant_documents": self.relevant_documents,
            "retry_count": self.retry_count,
            "query_rewritten": self.query_rewritten,
            "answer_generated": self.answer_generated,
            "retrieval_latency_ms": round(
                self.retrieval_latency_ms,
                2,
            ),
            "generation_latency_ms": round(
                self.generation_latency_ms,
                2,
            ),
            "total_latency_ms": round(
                self.total_latency_ms,
                2,
            ),
            "extra": self.extra,
        }
