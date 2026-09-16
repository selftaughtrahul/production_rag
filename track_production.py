"""Scan the repo and refresh PRODUCTION_TRACKER.md.

Run from the project root:

    python track_production.py
"""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

ROOT = Path(__file__).resolve().parent
Status = Literal["done", "partial", "todo"]

DONE = "done"
PARTIAL = "partial"
TODO = "todo"


def _read(rel: str) -> str:
    path = ROOT / rel
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _exists(rel: str) -> bool:
    return (ROOT / rel).exists()


_SEARCH_DIRS = ("app", "database", "frontend", "evals", "tests")
_SEARCH_ROOT_FILES = ("main.py", "run.py")


def _iter_source_files() -> list[Path]:
    files: list[Path] = []
    for name in _SEARCH_ROOT_FILES:
        path = ROOT / name
        if path.is_file():
            files.append(path)
    for folder in _SEARCH_DIRS:
        base = ROOT / folder
        if not base.is_dir():
            continue
        files.extend(path for path in base.rglob("*.py") if path.is_file())
    return files


def _any_source_contains(pattern: str) -> bool:
    for path in _iter_source_files():
        if pattern in path.read_text(encoding="utf-8", errors="replace"):
            return True
    return False


def _status(done: bool, partial: bool = False) -> Status:
    if done:
        return DONE
    if partial:
        return PARTIAL
    return TODO


@dataclass(frozen=True, slots=True)
class Check:
    id: str
    priority: str
    title: str
    why: str
    verify: Callable[[], Status]
    how_to_close: str


def check_jwt_auth() -> Status:
    auth = _read("app/api/auth.py")
    deps = _read("app/services/auth/dependencies.py")
    return _status("create_access_token" in auth and "get_current_user" in deps)


def check_session_owner() -> Status:
    return _status("ensure_session_owner" in _read("app/api/query.py"))


def check_tenant_filter() -> Status:
    nodes = _read("app/services/rag/nodes.py")
    return _status('metadata_filter = {"user_id": user_id}' in nodes)


def check_hybrid_rag() -> Status:
    hybrid = _read("app/services/retriever/hybrid.py")
    return _status("_rrf_fusion" in hybrid and "bm25_retriever" in hybrid)


def check_reranker() -> Status:
    return _status("CrossEncoderReranker" in _read("app/services/retriever/reranker.py"))


def check_supervisor() -> Status:
    graph = _read("app/agent/multi_agent/master_graph.py")
    return _status("supervisor" in graph and "rag_agent" in graph)


def check_sql_readonly() -> Status:
    sql = _read("app/tools/sql_search.py")
    return _status("_is_read_only" in sql and "DENIED_TABLES" in sql)


def check_input_length() -> Status:
    factory = _read("app/guardrails/factory.py")
    return _status("InputValidationGuardrail" in factory)


def check_jwt_secret() -> Status:
    config = _read("app/core/config.py")
    has_insecure_default = "change-this-secret-key-in-production" in config
    requires_key = "JWT_SECRET_KEY must be set" in config
    return _status(requires_key and not has_insecure_default, partial=not has_insecure_default)


def check_nested_thread_id() -> Status:
    nodes = _read("app/agent/multi_agent/nodes.py")
    invokes = "ainvoke(rag_input" in nodes or "ainvoke(rag_input)" in nodes
    passes_config = "configurable" in nodes and "thread_id" in nodes
    return _status(invokes and passes_config, partial=invokes)


def check_doc_search_text() -> Status:
    tool = _read("app/tools/doc_search.py")
    uses_page_content = "page_content" in tool
    uses_text = "doc.text" in tool
    return _status(uses_text and not uses_page_content, partial=uses_text)


def check_bm25_delete() -> Status:
    docs = _read("app/api/documents.py")
    store_has_delete = "def delete_document" in _read("app/services/retriever/bm25_store.py")
    api_calls_bm25 = "bm25" in docs.lower() and "delete" in docs
    return _status(store_has_delete and api_calls_bm25, partial=store_has_delete)


def check_health() -> Status:
    main = _read("main.py")
    return _status("/health" in main or '"health"' in main)


def check_logger_wired() -> Status:
    return _status("setup_logger" in _read("main.py"), partial=_exists("app/core/logger.py"))


def check_celery_compose() -> Status:
    compose = _read("docker-compose.yml")
    return _status("celery -A app.tasks.celery_app" in compose, partial="celery" in compose)


def check_llm_retry() -> Status:
    claude = _read("app/services/llm/claude.py")
    has_retry = "tenacity" in claude or "retry" in claude.lower() and "backoff" in claude.lower()
    has_timeout = "timeout" in claude
    return _status(bool(has_retry and has_timeout), partial="LLMUnavailableError" in claude)


def check_rerank_try_except() -> Status:
    rerank = _read("app/services/rag/nodes.py")
    fn = rerank.split("def rerank", 1)[-1].split("def grade_documents", 1)[0]
    return _status("try:" in fn and "except" in fn)


def check_nemo_fail_closed() -> Status:
    nemo = _read("app/guardrails/provider/nemo.py")
    fail_open = 'return {"passed": True, "content": query}' in nemo and "NeMo rail check failed" in nemo
    return _status(not fail_open, partial=True)


def check_output_guardrails_wired() -> Status:
    factory = _read("app/guardrails/factory.py")
    deps = _read("app/api/dependencies.py")
    wired = "presidio" in deps.lower() and "build_output_guardrails(" in deps
    always_none = "build_output_guardrails()" in deps
    return _status(wired and not always_none, partial="OutputGuardrailNode" in _read("app/agent/multi_agent/master_graph.py"))


def check_pii_wired() -> Status:
    deps = _read("app/api/dependencies.py")
    factory = _read("app/guardrails/factory.py")
    return _status(
        "Presidio" in deps or "presidio_provider=" in deps,
        partial="PresidioPIIGuardrail" in factory,
    )


def check_sse_stream() -> Status:
    query = _read("app/api/query.py")
    return _status("generate_stream" in query, partial="text/event-stream" in query)


def check_eval_harness() -> Status:
    in_app = _any_source_contains("from ragas") or _any_source_contains("import ragas")
    in_reqs = "ragas" in _read("requirements.txt")
    return _status(in_app, partial=in_reqs)


def check_tests() -> Status:
    tests = list(ROOT.glob("tests/**/*.py")) + list(ROOT.glob("test_*.py"))
    return _status(any(tests))


def check_cicd() -> Status:
    return _status(_exists(".github/workflows"))


def check_rate_limit() -> Status:
    return _status(
        _any_source_contains("SlowAPI") or _any_source_contains("Limiter(")
    )


def check_health_prod_docker() -> Status:
    compose = _read("docker-compose.yml")
    dockerfile = _read("Dockerfile")
    has_compose = bool(compose)
    prod = "--reload" not in compose and "USER " in dockerfile
    return _status(prod, partial=has_compose)


def check_postgres() -> Status:
    sqlite = _read("database/sqlite.py")
    uses_sqlite = "sqlite:///" in sqlite
    uses_pg = "postgresql" in sqlite or "postgres" in _read("app/core/config.py").lower()
    return _status(uses_pg and not uses_sqlite, partial=False)


def check_answer_cache() -> Status:
    query = _read("app/api/query.py")
    semantic = "semantic cache" in query.lower() or "response_cache" in query
    redis_cache = "redis" in query.lower() and "cache" in query.lower()
    return _status(semantic or redis_cache)


CHECKS: list[Check] = [
    Check("p0-jwt", "P0", "JWT register/login/me", "Identity for tenant isolation", check_jwt_auth, "Already in app/api/auth.py"),
    Check("p0-session", "P0", "Session ownership checks", "Stop users reading another thread", check_session_owner, "Already in app/api/session_access.py"),
    Check("p0-tenant", "P0", "Retrieval pre-filter by user_id", "No tenant leak in Chroma/BM25", check_tenant_filter, "Already in RAG retrieve node"),
    Check("p0-hybrid", "P0", "Hybrid RAG (dense + BM25 + RRF)", "Core retrieval quality", check_hybrid_rag, "Already in hybrid.py"),
    Check("p0-rerank", "P0", "Cross-encoder reranker", "Precision of context", check_reranker, "Already in reranker.py"),
    Check("p0-supervisor", "P0", "LangGraph supervisor + specialists", "Agentic routing", check_supervisor, "Already in master_graph.py"),
    Check("p0-sql-ro", "P0", "Read-only SQL tool + denied tables", "Tool safety", check_sql_readonly, "Already in sql_search.py"),
    Check("p0-input-len", "P0", "Input length validation", "Cheap abuse/cost cap", check_input_length, "Already in factory.py"),
    Check("p0-jwt-secret", "P0", "No insecure JWT secret default", "Stolen tokens if default ships", check_jwt_secret, "Require JWT_SECRET_KEY; raise if missing"),
    Check("p0-thread-id", "P0", "Nested RAG ainvoke passes thread_id", "Checkpoint correctness", check_nested_thread_id, "Pass config={'configurable': {'thread_id': ...}} in RAGSubGraphNode"),
    Check("p0-doc-text", "P0", "document_search uses .text", "Fallback RAG tool crash", check_doc_search_text, "Replace page_content with doc.text"),
    Check("p0-bm25-delete", "P0", "Delete BM25 chunks with the document", "Stale lexical hits after delete", check_bm25_delete, "Call BM25Store.delete_document in documents.py"),
    Check("p0-health", "P0", "GET /health", "Load balancer / Compose probes", check_health, "Add a FastAPI health route in main.py"),
    Check("p0-logger", "P0", "Wire JSON logger at startup", "Request traces in production", check_logger_wired, "Call setup_logger() from main.py"),
    Check("p0-celery-path", "P0", "Compose Celery -A app.tasks.celery_app", "Background ingest actually starts", check_celery_compose, "Fix docker-compose.yml worker command"),
    Check("p0-llm-retry", "P0", "LLM timeout + retry/backoff", "Quota/blips should not 500 the chat", check_llm_retry, "Add timeout and tenacity around Claude calls"),
    Check("p0-rerank-err", "P0", "Rerank node try/except + fallback", "Cross-encoder crash kills the turn", check_rerank_try_except, "Catch in RAGNodes.rerank; keep fused docs"),
    Check("p0-nemo", "P0", "NeMo injection: do not fail open", "Jailbreak allowed when NeMo errors", check_nemo_fail_closed, "On NeMo exception, block or use denylist only"),
    Check("p1-output-rails", "P1", "Output guardrails actually constructed", "Hallucination/PII at the exit door", check_output_guardrails_wired, "Pass providers into build_output_guardrails from dependencies.py"),
    Check("p1-pii", "P1", "Presidio PII wired", "PII in prompts/responses", check_pii_wired, "Instantiate PresidioProvider in factory/dependencies"),
    Check("p1-sse", "P1", "True SSE token streaming", "UI waits until full Claude answer", check_sse_stream, "Use ClaudeService.generate_stream in query.py"),
    Check("p1-eval", "P1", "Ragas/eval harness in repo", "Know if RAG got better", check_eval_harness, "Add evals/ with a golden set; call ragas from CI"),
    Check("p1-tests", "P1", "Automated tests", "Regressions on routing/SQL/auth", check_tests, "Add tests/ for auth, SQL allowlist, hybrid retrieve"),
    Check("p2-rate-limit", "P2", "API rate limiting", "Cost and abuse", check_rate_limit, "SlowAPI or gateway throttle per user_id"),
    Check("p2-cache", "P2", "Response/semantic cache", "Repeat-question cost", check_answer_cache, "Redis keyed by user + query hash"),
    Check("p3-docker-prod", "P3", "Prod Compose (no --reload, non-root)", "Safe container run", check_health_prod_docker, "Drop --reload; USER in Dockerfile"),
    Check("p3-cicd", "P3", "GitHub Actions CI", "Lint/test before merge", check_cicd, "Add .github/workflows/ci.yml"),
    Check("p3-postgres", "P3", "Postgres instead of SQLite for app state", "Multi-instance API", check_postgres, "Move users/memories/checkpoints to RDS"),
]


def run_checks() -> list[tuple[Check, Status]]:
    return [(item, item.verify()) for item in CHECKS]


def _mark(status: Status) -> str:
    return {"done": "✅", "partial": "🟡", "todo": "⬜"}[status]


def _counts(rows: list[tuple[Check, Status]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for check, status in rows:
        bucket = out.setdefault(check.priority, {"done": 0, "partial": 0, "todo": 0, "total": 0})
        bucket[status] += 1
        bucket["total"] += 1
    return out


def render_markdown(rows: list[tuple[Check, Status]]) -> str:
    now = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    by_pri = _counts(rows)
    done = sum(1 for _, s in rows if s == DONE)
    total = len(rows)
    pct = round(100 * done / total) if total else 0

    lines = [
        "# Production tracker",
        "",
        "This file is **generated**. Do not edit it by hand.",
        "",
        f"- Last scan: `{now}`",
        f"- Command: `python track_production.py`",
        f"- Closed items: **{done}/{total} ({pct}%)**",
        "",
        "Status: ✅ done (code evidence) · 🟡 partial · ⬜ not done",
        "",
        "## Scoreboard",
        "",
        "| Priority | Done | Partial | Todo | Total |",
        "| -------- | ---- | ------- | ---- | ----- |",
    ]
    for pri in ("P0", "P1", "P2", "P3"):
        b = by_pri.get(pri, {"done": 0, "partial": 0, "todo": 0, "total": 0})
        lines.append(f"| {pri} | {b['done']} | {b['partial']} | {b['todo']} | {b['total']} |")

    lines += [
        "",
        "## How we use this",
        "",
        "1. Run `python track_production.py` after a production change.",
        "2. Pick the next **⬜ P0** row and implement it.",
        "3. Re-run the tracker. The row should flip to ✅.",
        "4. Do not call the project production-ready until **P0 is all ✅** and P1 eval exists.",
        "",
    ]

    current_pri = ""
    for check, status in rows:
        if check.priority != current_pri:
            current_pri = check.priority
            lines += ["", f"## {current_pri}", "", "| Status | ID | Item | Why | How to close |", "| ------ | -- | ---- | --- | ------------ |"]
        lines.append(
            f"| {_mark(status)} | `{check.id}` | {check.title} | {check.why} | {check.how_to_close} |"
        )

    next_open = next((c for c, s in rows if s != DONE), None)
    lines += ["", "## Next action", ""]
    if next_open is None:
        lines.append("All tracked items are ✅. Revisit AWS/deploy SLOs outside this list.")
    else:
        lines.append(f"Work next: **{next_open.id} — {next_open.title}**")
        lines.append("")
        lines.append(next_open.how_to_close)

    lines += ["", "---", "", "*Generated by `track_production.py`. Code is the source of truth.*", ""]
    return "\n".join(lines)


def _mark_ascii(status: Status) -> str:
    return {"done": "[x]", "partial": "[~]", "todo": "[ ]"}[status]


def print_report(rows: list[tuple[Check, Status]]) -> None:
    done = sum(1 for _, s in rows if s == DONE)
    print(f"Production tracker  {done}/{len(rows)} done")
    print("")
    current = ""
    for check, status in rows:
        if check.priority != current:
            current = check.priority
            print(f"  {current}")
        print(f"    {_mark_ascii(status)}  {check.id:18} {check.title}")
    nxt = next((c for c, s in rows if s != DONE), None)
    print("")
    if nxt:
        print(f"Next: {nxt.id} - {nxt.title}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan the repo and refresh PRODUCTION_TRACKER.md")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if any P0 item is not fully done (CI gate).",
    )
    args = parser.parse_args()

    rows = run_checks()
    tracker = ROOT / "PRODUCTION_TRACKER.md"
    tracker.write_text(render_markdown(rows), encoding="utf-8")
    print_report(rows)
    print(f"\nWrote {tracker}")

    if args.check:
        p0_open = [c.id for c, s in rows if c.priority == "P0" and s != DONE]
        if p0_open:
            print("P0 still open: " + ", ".join(p0_open))
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
