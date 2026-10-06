import logging
import os
import sys
from pathlib import Path
from typing import Protocol

from celery import Celery
from celery.signals import task_prerun

logger = logging.getLogger(__name__)

# Celery puts the project folder on sys.path only while loading this module,
# then removes it. Insert it again immediately before each task runs.
ROOT = Path(__file__).resolve().parents[2]


def _ensure_project_root(**_kwargs: object) -> None:
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_project_root()
task_prerun.connect(_ensure_project_root)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "basic_rag",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["app.tasks.tasks"],
)

celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

def summarize_inspect(
    active: dict | None,
    reserved: dict | None,
    scheduled: dict | None,
) -> dict[str, int | None]:
    """Turn Celery inspect maps into worker and task counts."""

    def count(payload: dict | None) -> int | None:
        if payload is None:
            return None
        return sum(len(tasks) for tasks in payload.values())

    workers = None if active is None else len(active)
    return {
        "workers": workers,
        "active": count(active),
        "reserved": count(reserved),
        "scheduled": count(scheduled),
    }


class BrokerClient(Protocol):
    def ping(self) -> object: ...

    def llen(self, name: str) -> int: ...


def broker_waiting(client: BrokerClient) -> int | None:
    """Jobs sitting in the default Celery Redis list. None when Redis is down."""
    try:
        client.ping()
        return int(client.llen("celery"))
    except Exception:
        logger.warning("Ingest queue broker is unreachable", exc_info=True)
        return None


if __name__ == "__main__":
    celery_app.start()
