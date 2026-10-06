"""Dashboard GPU and ingest-queue status."""

from __future__ import annotations

import logging
import os
import time
from threading import Lock
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.device import gpu_snapshot
from app.models.schemas import GpuStatus, MonitorStatus, QueueItem, QueueStatus, UserInDB
from app.services.auth.dependencies import get_current_user
from app.tasks.celery_app import broker_waiting, celery_app, summarize_inspect
from database.models import Document
from database.sqlite import get_db

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ops"])

_INSPECT_TTL_SECONDS = 4.0
_inspect_lock = Lock()
_inspect_cached: tuple[float, dict[str, int | None]] | None = None


def _redis_client() -> Any:
    import redis

    url = os.getenv("REDIS_URL", "redis://localhost:6379/0").strip()
    return redis.Redis.from_url(
        url,
        socket_connect_timeout=0.4,
        socket_timeout=0.4,
    )


def worker_counts() -> dict[str, int | None]:
    """Active Celery tasks, cached so the dashboard poll does not fan out."""
    global _inspect_cached
    now = time.monotonic()
    with _inspect_lock:
        if _inspect_cached is not None and now - _inspect_cached[0] < _INSPECT_TTL_SECONDS:
            return dict(_inspect_cached[1])
    try:
        inspector = celery_app.control.inspect(timeout=0.5)
        summary = summarize_inspect(inspector.active(), None, None)
    except Exception:
        logger.warning("Celery inspect failed", exc_info=True)
        summary = summarize_inspect(None, None, None)
    with _inspect_lock:
        _inspect_cached = (now, summary)
    return summary


def document_queue(db: Session, user_id: str) -> tuple[int, int, list[QueueItem]]:
    """This user's ingest documents that are still processing or have failed."""
    rows = db.execute(
        select(Document.status, func.count())
        .where(Document.user_id == user_id)
        .group_by(Document.status)
    ).all()
    counts = {status: int(count) for status, count in rows}
    processing = db.scalars(
        select(Document)
        .where(Document.user_id == user_id, Document.status == "processing")
        .order_by(Document.created_at.desc())
        .limit(8)
    ).all()
    items = [
        QueueItem(id=row.id, file_name=row.file_name, status="processing")
        for row in processing
    ]
    return counts.get("processing", 0), counts.get("failed", 0), items


@router.get("/monitor", response_model=MonitorStatus)
def monitor_status(
    current_user: UserInDB = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MonitorStatus:
    """GPU plus the ingest queue for the signed-in user."""
    waiting: int | None
    try:
        waiting = broker_waiting(_redis_client())
    except Exception:
        logger.warning("Could not open the ingest queue broker", exc_info=True)
        waiting = None
    inspect = worker_counts() if waiting is not None else summarize_inspect(None, None, None)
    processing, failed, items = document_queue(db, current_user.id)
    return MonitorStatus(
        gpu=GpuStatus.model_validate(gpu_snapshot()),
        queue=QueueStatus(
            broker_ok=waiting is not None,
            waiting=waiting,
            workers=inspect["workers"],
            active=inspect["active"],
            processing=processing,
            failed=failed,
            items=items,
        ),
    )
