# Production tracker

This file is **generated**. Do not edit it by hand.

- Last scan: `2026-09-16 23:56`
- Command: `python track_production.py`
- Closed items: **36/41 (88%)**

Status: ✅ done (code evidence) · 🟡 partial · ⬜ not done

## Scoreboard

| Priority | Done | Partial | Todo | Total |
| -------- | ---- | ------- | ---- | ----- |
| P0 | 18 | 0 | 0 | 18 |
| P1 | 5 | 0 | 0 | 5 |
| P2 | 2 | 0 | 0 | 2 |
| P3 | 3 | 0 | 0 | 3 |
| P4 | 8 | 0 | 5 | 13 |

## How we use this

1. Run `python track_production.py` after a production change.
2. Pick the next **⬜ P0** row and implement it.
3. Re-run the tracker. The row should flip to ✅.
4. Do not call the project production-ready until **P0 is all ✅** and P1 eval exists.


## P0

| Status | ID | Item | Why | How to close |
| ------ | -- | ---- | --- | ------------ |
| ✅ | `p0-jwt` | JWT register/login/me | Identity for tenant isolation | Already in app/api/auth.py |
| ✅ | `p0-session` | Session ownership checks | Stop users reading another thread | Already in app/api/session_access.py |
| ✅ | `p0-tenant` | Retrieval pre-filter by user_id | No tenant leak in Chroma/BM25 | Already in RAG retrieve node |
| ✅ | `p0-hybrid` | Hybrid RAG (dense + BM25 + RRF) | Core retrieval quality | Already in hybrid.py |
| ✅ | `p0-rerank` | Cross-encoder reranker | Precision of context | Already in reranker.py |
| ✅ | `p0-supervisor` | LangGraph supervisor + specialists | Agentic routing | Already in master_graph.py |
| ✅ | `p0-sql-ro` | Read-only SQL tool + denied tables | Tool safety | Already in sql_search.py |
| ✅ | `p0-input-len` | Input length validation | Cheap abuse/cost cap | Already in factory.py |
| ✅ | `p0-jwt-secret` | No insecure JWT secret default | Stolen tokens if default ships | Require JWT_SECRET_KEY; raise if missing |
| ✅ | `p0-thread-id` | Nested RAG ainvoke passes thread_id | Checkpoint correctness | Pass config={'configurable': {'thread_id': ...}} in RAGSubGraphNode |
| ✅ | `p0-doc-text` | document_search uses .text | Fallback RAG tool crash | Replace page_content with doc.text |
| ✅ | `p0-bm25-delete` | Delete BM25 chunks with the document | Stale lexical hits after delete | Call BM25Store.delete_document in documents.py |
| ✅ | `p0-health` | GET /health | Load balancer / Compose probes | Add a FastAPI health route in main.py |
| ✅ | `p0-logger` | Wire JSON logger at startup | Request traces in production | Call setup_logger() from main.py |
| ✅ | `p0-celery-path` | Compose Celery -A app.tasks.celery_app | Background ingest actually starts | Fix docker-compose.yml worker command |
| ✅ | `p0-llm-retry` | LLM timeout + retry/backoff | Quota/blips should not 500 the chat | Add timeout and tenacity around Claude calls |
| ✅ | `p0-rerank-err` | Rerank node try/except + fallback | Cross-encoder crash kills the turn | Catch in RAGNodes.rerank; keep fused docs |
| ✅ | `p0-nemo` | NeMo injection: do not fail open | Jailbreak allowed when NeMo errors | On NeMo exception, block or use denylist only |

## P1

| Status | ID | Item | Why | How to close |
| ------ | -- | ---- | --- | ------------ |
| ✅ | `p1-output-rails` | Output guardrails actually constructed | Hallucination/PII at the exit door | Pass providers into build_output_guardrails from dependencies.py |
| ✅ | `p1-pii` | Presidio PII wired | PII in prompts/responses | Instantiate PresidioProvider in factory/dependencies |
| ✅ | `p1-sse` | True SSE token streaming | UI waits until full Claude answer | Use ClaudeService.generate_stream in query.py |
| ✅ | `p1-eval` | Ragas/eval harness in repo | Know if RAG got better | Add evals/ with a golden set; call ragas from CI |
| ✅ | `p1-tests` | Automated tests | Regressions on routing/SQL/auth | Add tests/ for auth, SQL allowlist, hybrid retrieve |

## P2

| Status | ID | Item | Why | How to close |
| ------ | -- | ---- | --- | ------------ |
| ✅ | `p2-rate-limit` | API rate limiting | Cost and abuse | SlowAPI or gateway throttle per user_id |
| ✅ | `p2-cache` | Response/semantic cache | Repeat-question cost | Redis keyed by user + query hash |

## P3

| Status | ID | Item | Why | How to close |
| ------ | -- | ---- | --- | ------------ |
| ✅ | `p3-docker-prod` | Prod Compose (no --reload, non-root) | Safe container run | Drop --reload; USER in Dockerfile |
| ✅ | `p3-cicd` | GitHub Actions CI | Lint/test before merge | Add .github/workflows/ci.yml |
| ✅ | `p3-postgres` | Postgres instead of SQLite for app state | Multi-instance API | Move users/memories/checkpoints to RDS |

## P4

| Status | ID | Item | Why | How to close |
| ------ | -- | ---- | --- | ------------ |
| ✅ | `p4-audit` | Audit logs for chat and tools | Replay and abuse review | audit_event on chat/tool calls; no raw PII in logs |
| ✅ | `p4-prometheus` | Prometheus /metrics | RED metrics for the API | GET /metrics Prometheus text |
| ✅ | `p4-mcp` | MCP server for RAG | Cursor and other MCP clients | app/mcp/server.py tools/list + ask_rag |
| ✅ | `p4-jailbreak` | Jailbreak detection | Bypass of system policy | JailbreakGuardrail on input (denylist + NeMo) |
| ✅ | `p4-human-approval` | Human approval gate on write tools | Irreversible agent actions | requires_approval on BaseAgentTool; invoke_tool refuses until approved |
| ✅ | `p4-langsmith` | LangSmith tracing wired at startup | Distributed LLM traces | Set LANGCHAIN_TRACING_V2 from Settings in main.py |
| ✅ | `p4-structured` | Structured LLM outputs | Supervisor JSON schema | ClaudeService.agenerate_structured + Pydantic |
| ✅ | `p4-memory` | Short/long/conversation/episodic memory | Agent memory types | Checkpoints + user_memories + episodic/semantic types |
| ⬜ | `p4-deepeval` | DeepEval harness | Alternate LLM-as-judge | Add evals using deepeval (Ragas already covers judge) |
| ⬜ | `p4-langfuse` | Langfuse tracing | Open-source LLM observability | Optional Langfuse client; LangSmith is the current tracer |
| ⬜ | `p4-mlflow` | MLflow experiment tracking | Prompt/model experiment store | Add MLflow only if you need experiment registry beyond LangSmith |
| ⬜ | `p4-grafana` | Grafana dashboards | Human metrics UI | Point Grafana at /metrics; not shipped in Compose yet |
| ⬜ | `p4-knowledge-graph` | Knowledge graphs | Entity/relation RAG | Not in this stack; hybrid Chroma+BM25 is the knowledge store |

## Next action

Work next: **p4-deepeval — DeepEval harness**

Add evals using deepeval (Ragas already covers judge)

---

*Generated by `track_production.py`. Code is the source of truth.*
