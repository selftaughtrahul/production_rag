import json
import logging
import os
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any

LOG_DIR = os.getenv("LOG_DIR", "logs")

MAX_LOG_SIZE = 50 * 1024 * 1024  # 50 MB
BACKUP_COUNT = 10


class JSONFormatter(logging.Formatter):

    def format(self, record: logging.LogRecord) -> str:

        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add structured fields
        if hasattr(record, "event"):
            log_data["event"] = record.event

        if hasattr(record, "run_id"):
            log_data["run_id"] = record.run_id

        if hasattr(record, "session_id"):
            log_data["session_id"] = record.session_id

        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id

        if hasattr(record, "node"):
            log_data["node"] = record.node

        if hasattr(record, "latency_ms"):
            log_data["latency_ms"] = record.latency_ms

        if hasattr(record, "data"):
            log_data["data"] = record.data

        if record.exc_info:
            log_data["exception"] = self.formatException(
                record.exc_info
            )

        return json.dumps(
            log_data,
            ensure_ascii=False,
            default=str,
        )


def setup_logger() -> logging.Logger:

    logger = logging.getLogger("rag")

    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    formatter = JSONFormatter()

    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        rag_handler = RotatingFileHandler(
            filename=os.path.join(LOG_DIR, "rag.log"),
            maxBytes=MAX_LOG_SIZE,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        rag_handler.setLevel(logging.INFO)
        rag_handler.setFormatter(formatter)
        logger.addHandler(rag_handler)

        error_handler = RotatingFileHandler(
            filename=os.path.join(LOG_DIR, "error.log"),
            maxBytes=MAX_LOG_SIZE,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)
    except OSError:
        pass

    console_handler = logging.StreamHandler(sys.stdout)

    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.propagate = False

    return logger


logger = setup_logger()


def log_event(
    event: str,
    *,
    level: str = "info",
    run_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    node: str | None = None,
    latency_ms: float | None = None,
    data: dict[str, Any] | None = None,
    message: str = "",
):

    extra = {
        "event": event,
        "run_id": run_id,
        "session_id": session_id,
        "user_id": user_id,
        "node": node,
        "latency_ms": latency_ms,
        "data": data or {},
    }

    log_method = getattr(logger, level, logger.info)

    log_method(
        message or event,
        extra=extra,
    )