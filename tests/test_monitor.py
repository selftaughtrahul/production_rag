"""GPU and ingest-queue snapshots used by the dashboard."""

from __future__ import annotations

from app.core.device import parse_gpu_csv
from app.tasks.celery_app import broker_waiting, summarize_inspect


def test_parse_gpu_csv_reads_utilization_memory_and_name() -> None:
    parsed = parse_gpu_csv("17, 2048, 8192, NVIDIA GeForce RTX 3060, 61\n")
    assert parsed is not None
    assert parsed["available"] is True
    assert parsed["name"] == "NVIDIA GeForce RTX 3060"
    assert parsed["utilization_percent"] == 17
    assert parsed["memory_used_mb"] == 2048
    assert parsed["memory_total_mb"] == 8192
    assert parsed["temperature_c"] == 61


def test_parse_gpu_csv_rejects_blank_output() -> None:
    assert parse_gpu_csv("") is None
    assert parse_gpu_csv("n/a, n/a, n/a, , n/a") is None


def test_summarize_inspect_counts_workers_and_tasks() -> None:
    summary = summarize_inspect(
        active={"worker-a": [{"id": "1"}], "worker-b": []},
        reserved={"worker-a": [{"id": "2"}, {"id": "3"}]},
        scheduled=None,
    )
    assert summary["workers"] == 2
    assert summary["active"] == 1
    assert summary["reserved"] == 2
    assert summary["scheduled"] is None


def test_summarize_inspect_is_unknown_when_no_workers_reply() -> None:
    summary = summarize_inspect(None, None, None)
    assert summary == {
        "workers": None,
        "active": None,
        "reserved": None,
        "scheduled": None,
    }


class _Broker:
    def __init__(self, depth: int) -> None:
        self.depth = depth

    def ping(self) -> str:
        return "PONG"

    def llen(self, name: str) -> int:
        assert name == "celery"
        return self.depth


def test_broker_waiting_reads_celery_list_length() -> None:
    assert broker_waiting(_Broker(4)) == 4


def test_broker_waiting_is_none_when_ping_fails() -> None:
    class _Down:
        def ping(self) -> str:
            raise OSError("redis down")

        def llen(self, name: str) -> int:
            raise AssertionError("llen should not run")

    assert broker_waiting(_Down()) is None
