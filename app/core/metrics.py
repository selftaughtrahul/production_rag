"""Prometheus text exposition for /metrics (no extra scrape client required)."""

from __future__ import annotations

from collections import defaultdict
from threading import Lock

_lock = Lock()
_counters: dict[tuple[str, str], int] = defaultdict(int)

# Ingestion totals keyed by (parser, document_type); summed, not per-document.
_INGEST_FIELDS = (
    "documents",
    "pages",
    "elements",
    "tables",
    "chunks",
    "failed_pages",
    "ocr_documents",
    "fallback_documents",
)
_ingest: dict[tuple[str, str], dict[str, float]] = defaultdict(
    lambda: {name: 0.0 for name in (*_INGEST_FIELDS, "seconds")}
)


def inc_chat(outcome: str) -> None:
    """Count chat completions by outcome: ok, error, blocked, cache."""
    key = ("rag_chat_requests_total", outcome)
    with _lock:
        _counters[key] += 1


def record_ingestion(
    *,
    parser: str,
    document_type: str,
    page_count: int | None,
    element_count: int,
    table_count: int,
    chunk_count: int,
    failed_pages: int,
    ocr_used: bool,
    fell_back: bool,
    processing_seconds: float,
) -> None:
    """Accumulate one ingested document into the parser totals."""
    key = (parser or "unknown", document_type or "unknown")
    with _lock:
        bucket = _ingest[key]
        bucket["documents"] += 1
        bucket["pages"] += page_count or 0
        bucket["elements"] += element_count
        bucket["tables"] += table_count
        bucket["chunks"] += chunk_count
        bucket["failed_pages"] += failed_pages
        bucket["ocr_documents"] += 1 if ocr_used else 0
        bucket["fallback_documents"] += 1 if fell_back else 0
        bucket["seconds"] += processing_seconds


def _label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def render_prometheus() -> bytes:
    """Return Prometheus text format for HTTP /metrics."""
    lines = [
        "# HELP rag_chat_requests_total Chat POSTs by outcome",
        "# TYPE rag_chat_requests_total counter",
    ]
    with _lock:
        items = list(_counters.items())
        ingest_items = [(key, dict(value)) for key, value in _ingest.items()]
    if not items:
        lines.append('rag_chat_requests_total{outcome="ok"} 0')
    else:
        for (_, outcome), value in sorted(items, key=lambda item: item[0][1]):
            lines.append(f'rag_chat_requests_total{{outcome="{outcome}"}} {value}')

    for name in _INGEST_FIELDS:
        metric = f"rag_ingest_{name}_total"
        lines.append(f"# HELP {metric} Ingestion {name.replace('_', ' ')} by parser and document type")
        lines.append(f"# TYPE {metric} counter")
        for (parser, doc_type), bucket in sorted(ingest_items):
            lines.append(
                f'{metric}{{parser="{_label(parser)}",document_type="{_label(doc_type)}"}} '
                f"{int(bucket[name])}"
            )
    lines.append("# HELP rag_ingest_seconds_total Parser wall time by parser and document type")
    lines.append("# TYPE rag_ingest_seconds_total counter")
    for (parser, doc_type), bucket in sorted(ingest_items):
        lines.append(
            f'rag_ingest_seconds_total{{parser="{_label(parser)}",document_type="{_label(doc_type)}"}} '
            f"{bucket['seconds']:.6f}"
        )
    return ("\n".join(lines) + "\n").encode("utf-8")
