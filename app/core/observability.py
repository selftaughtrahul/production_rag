

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any
from core.config import Settings


config = Settings()

LOG_DIR = config.LOG_DIR
MAX_LOG_SIZE = 50 * 1024 * 1024       # 50 MB
BACKUP_COUNT = 10

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


# ============================================================
# JSON FORMATTER
# ============================================================

class JSONFormatter(logging.Formatter):

    def format(self, record: logging.LogRecord) -> str:

        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Structured fields
        fields = [
            "event",
            "run_id",
            "trace_id",
            "session_id",
            "user_id",
            "node",
            "latency_ms",
            "status",
            "data",
            "metrics",
        ]

        for field_name in fields:

            value = getattr(record, field_name, None)

            if value is not None:
                log_data[field_name] = value

        # Exception
        if record.exc_info:

            log_data["exception"] = self.formatException(
                record.exc_info
            )

        return json.dumps(
            log_data,
            ensure_ascii=False,
            default=str,
        )


# ============================================================
# LOGGER SETUP
# ============================================================

def setup_logger() -> logging.Logger:

    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger("rag")

    logger.setLevel(LOG_LEVEL)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    formatter = JSONFormatter()

    # --------------------------------------------------------
    # MAIN RAG LOG
    # --------------------------------------------------------

    rag_handler = RotatingFileHandler(
        filename=os.path.join(LOG_DIR, "rag.log"),
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )

    rag_handler.setLevel(logging.INFO)
    rag_handler.setFormatter(formatter)

    # --------------------------------------------------------
    # ERROR LOG
    # --------------------------------------------------------

    error_handler = RotatingFileHandler(
        filename=os.path.join(LOG_DIR, "error.log"),
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )

    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    # --------------------------------------------------------
    # CONSOLE
    # --------------------------------------------------------

    console_handler = logging.StreamHandler(sys.stdout)

    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # --------------------------------------------------------
    # REGISTER
    # --------------------------------------------------------

    logger.addHandler(rag_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)

    logger.propagate = False

    return logger


logger = setup_logger()


# ============================================================
# OBSERVABILITY CONTEXT
# ============================================================

@dataclass
class ObservabilityContext:

    run_id: str | None = None
    trace_id: str | None = None
    session_id: str | None = None
    user_id: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# OBSERVABILITY SERVICE
# ============================================================

class ObservabilityService:
    """
    Central observability service for the RAG pipeline.

    Responsible for:

        - Structured JSON logging
        - Node execution tracking
        - Latency tracking
        - Error tracking
        - Metrics
        - Evaluation data collection
        - Future Prometheus integration
        - Future Ragas integration
        - Future DeepEval integration
        - LangSmith correlation
    """

    def __init__(
        self,
        context: ObservabilityContext | None = None,
    ):

        self.context = (
            context
            or ObservabilityContext()
        )

    # ========================================================
    # INTERNAL LOG METHOD
    # ========================================================

    def _log(
        self,
        event: str,
        *,
        level: str = "info",
        node: str | None = None,
        latency_ms: float | None = None,
        status: str | None = None,
        data: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        message: str = "",
        exc_info: bool = False,
    ):

        extra = {

            "event": event,

            "run_id":
                self.context.run_id,

            "trace_id":
                self.context.trace_id,

            "session_id":
                self.context.session_id,

            "user_id":
                self.context.user_id,

            "node": node,

            "latency_ms": latency_ms,

            "status": status,

            "data": data or {},

            "metrics": metrics or {},
        }

        log_method = getattr(
            logger,
            level,
            logger.info,
        )

        log_method(
            message or event,
            extra=extra,
            exc_info=exc_info,
        )

    # ========================================================
    # RUN START
    # ========================================================

    def run_start(
        self,
        *,
        data: dict[str, Any] | None = None,
    ):

        self._log(
            "run_started",
            data=data,
            status="started",
        )

    # ========================================================
    # RUN END
    # ========================================================

    def run_end(
        self,
        *,
        latency_ms: float,
        data: dict[str, Any] | None = None,
    ):

        self._log(
            "run_completed",
            latency_ms=latency_ms,
            data=data,
            status="success",
        )

    # ========================================================
    # NODE START
    # ========================================================

    def node_start(
        self,
        node: str,
        *,
        data: dict[str, Any] | None = None,
    ):

        self._log(
            "node_started",
            node=node,
            data=data,
            status="started",
        )

    # ========================================================
    # NODE END
    # ========================================================

    def node_end(
        self,
        node: str,
        *,
        latency_ms: float,
        data: dict[str, Any] | None = None,
    ):

        self._log(
            "node_completed",
            node=node,
            latency_ms=latency_ms,
            data=data,
            status="success",
        )

    # ========================================================
    # NODE ERROR
    # ========================================================

    def node_error(
        self,
        node: str,
        *,
        latency_ms: float | None = None,
        error: Exception | None = None,
        data: dict[str, Any] | None = None,
    ):

        if error:

            data = {
                **(data or {}),
                "error_type": type(error).__name__,
                "error_message": str(error),
            }

        self._log(
            "node_failed",
            level="error",
            node=node,
            latency_ms=latency_ms,
            data=data,
            status="failed",
            message=str(error) if error else "Node failed",
            exc_info=bool(error),
        )

    # ========================================================
    # GENERIC ERROR
    # ========================================================

    def error(
        self,
        event: str,
        error: Exception,
        *,
        node: str | None = None,
        data: dict[str, Any] | None = None,
    ):

        data = {
            **(data or {}),
            "error_type": type(error).__name__,
            "error_message": str(error),
        }

        self._log(
            event,
            level="error",
            node=node,
            data=data,
            status="failed",
            message=str(error),
            exc_info=True,
        )

    # ========================================================
    # METRICS
    # ========================================================

    def metric(
        self,
        name: str,
        value: float,
        *,
        node: str | None = None,
        labels: dict[str, str] | None = None,
    ):

        self._log(
            "metric",
            node=node,
            metrics={
                "name": name,
                "value": value,
                "labels": labels or {},
            },
        )

    # ========================================================
    # TOKEN USAGE
    # ========================================================

    def token_usage(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int | None = None,
        node: str | None = None,
        model: str | None = None,
    ):

        if total_tokens is None:

            total_tokens = (
                input_tokens +
                output_tokens
            )

        self._log(
            "llm_token_usage",
            node=node,
            metrics={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            },
            data={
                "model": model,
            },
        )

    # ========================================================
    # RETRIEVAL
    # ========================================================

    def retrieval(
        self,
        *,
        query: str,
        documents_count: int,
        node: str = "retrieve",
        latency_ms: float | None = None,
    ):

        self._log(
            "retrieval_completed",
            node=node,
            latency_ms=latency_ms,
            data={
                "query": query,
                "documents_count": documents_count,
            },
        )

        self.metric(
            "retrieved_documents",
            documents_count,
            node=node,
        )

    # ========================================================
    # GENERATION
    # ========================================================

    def generation(
        self,
        *,
        model: str,
        latency_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        node: str = "generate",
    ):

        self._log(
            "generation_completed",
            node=node,
            latency_ms=latency_ms,
            data={
                "model": model,
            },
            metrics={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens":
                    input_tokens + output_tokens,
            },
        )

    # ========================================================
    # EVALUATION DATA
    # ========================================================

    def evaluation(
        self,
        *,
        question: str,
        answer: str,
        contexts: list[str] | None = None,
        reference: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Stores evaluation-ready data.

        This data can later be consumed by:

            Ragas
            DeepEval
        """

        self._log(
            "evaluation_record",
            data={
                "question": question,
                "answer": answer,
                "contexts": contexts or [],
                "reference": reference,
                "metadata": metadata or {},
            },
        )

    # ========================================================
    # CUSTOM EVENT
    # ========================================================

    def event(
        self,
        event: str,
        *,
        node: str | None = None,
        data: dict[str, Any] | None = None,
        status: str | None = None,
    ):

        self._log(
            event,
            node=node,
            data=data,
            status=status,
        )

    # ========================================================
    # TIMER
    # ========================================================

    def timer(self):

        return time.perf_counter()


# ============================================================
# FACTORY
# ============================================================

def create_observability(
    *,
    run_id: str | None = None,
    trace_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> ObservabilityService:

    context = ObservabilityContext(

        run_id=run_id,

        trace_id=trace_id,

        session_id=session_id,

        user_id=user_id,
    )

    return ObservabilityService(
        context=context
    )
