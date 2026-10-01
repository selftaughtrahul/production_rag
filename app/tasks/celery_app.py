import os
import sys
from pathlib import Path

from celery import Celery
from celery.signals import task_prerun

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

if __name__ == "__main__":
    celery_app.start()
