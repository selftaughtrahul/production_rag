from __future__ import annotations

import time

from typing import Any

from .metrics import RAGMetrics


class RAGObserver:
    """
    Helper used to observe a single RAG request.
    """

    def __init__(self) -> None:

        self.metrics = RAGMetrics()

    def on_retrieval_start(self) -> float:
        """
        Start retrieval timer.
        """

        return time.perf_counter()

    def on_retrieval_end(
        self,
        start_time: float,
        document_count: int,
    ) -> None:
        """
        Record retrieval completion.
        """

        latency_ms = (time.perf_counter() - start_time) * 1000

        self.metrics.record_retrieval(
            document_count=document_count,
            latency_ms=latency_ms,
        )

    def on_documents_graded(
        self,
        relevant_count: int,
    ) -> None:
        """
        Record document grading.
        """

        self.metrics.record_grading(relevant_count)

    def on_query_rewritten(self) -> None:
        """
        Record query rewriting.
        """

        self.metrics.record_rewrite()

    def on_retry(self) -> None:
        """
        Record a corrective RAG retry.
        """

        self.metrics.record_retry()

    def on_generation_start(self) -> float:
        """
        Start answer generation timer.
        """

        return time.perf_counter()

    def on_generation_end(
        self,
        start_time: float,
    ) -> None:
        """
        Record answer generation completion.
        """

        latency_ms = (time.perf_counter() - start_time) * 1000

        self.metrics.record_generation(latency_ms=latency_ms)

    def set_metadata(
        self,
        key: str,
        value: Any,
    ) -> None:
        """
        Add custom metadata.
        """

        self.metrics.set_extra(
            key,
            value,
        )

    def finish(self) -> dict[str, Any]:
        """
        Finalize and return request metrics.
        """

        return self.metrics.finish_and_get()
