"""Prometheus text exposition for /metrics (no extra scrape client required)."""

from __future__ import annotations

from collections import defaultdict
from threading import Lock

_lock = Lock()
_counters: dict[tuple[str, str], int] = defaultdict(int)


def inc_chat(outcome: str) -> None:
    """Count chat completions by outcome: ok, error, blocked, cache."""
    key = ("rag_chat_requests_total", outcome)
    with _lock:
        _counters[key] += 1


def render_prometheus() -> bytes:
    """Return Prometheus text format for HTTP /metrics."""
    lines = [
        "# HELP rag_chat_requests_total Chat POSTs by outcome",
        "# TYPE rag_chat_requests_total counter",
    ]
    with _lock:
        items = list(_counters.items())
    if not items:
        lines.append('rag_chat_requests_total{outcome="ok"} 0')
    else:
        for (_, outcome), value in sorted(items, key=lambda item: item[0][1]):
            lines.append(f'rag_chat_requests_total{{outcome="{outcome}"}} {value}')
    return ("\n".join(lines) + "\n").encode("utf-8")
