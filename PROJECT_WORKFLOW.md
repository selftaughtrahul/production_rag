# basic_rag — how the product works

This file is the operator’s map of the **current** codebase. It is not a roadmap. Chat has **one** pipeline: a LangGraph supervisor over hybrid retrieval. There is no `/query` mode switch and no second RAG path.

Session ids are always `session_<uuid>`.

---

## 1. What you are looking at

A user talks to Streamlit. Streamlit calls FastAPI with a JWT. FastAPI runs a **master LangGraph**. The supervisor picks a specialist (RAG, SQL, web, or general). The RAG specialist runs a **nested CRAG graph**: dense Chroma + BM25 FTS5 → RRF → knowledge-graph boost → cross-encoder → grade → optional rewrite → context. Claude then **token-streams** the answer over SSE. Guardrails, cache, rate limits, audit logs, and optional traces wrap that turn.

```mermaid
flowchart LR
    subgraph clients [Clients]
        UI[Streamlit UI]
        MCP[MCP stdio ask_rag]
        HTTP[curl / OpenAPI]
    end

    subgraph api [FastAPI main.py]
        Auth["/auth"]
        Docs["/ingest /documents"]
        Chat["POST /chat/ SSE"]
        Ops["/health /metrics"]
    end

    subgraph graphs [LangGraph]
        Master[Master supervisor]
        Nested[Nested RAG CRAG]
    end

    subgraph stores [State and indexes]
        PG[(Postgres users memories checkpoints)]
        Chroma[(Chroma vectors)]
        BM25[(SQLite FTS5 BM25)]
        Redis[(Redis cache + Celery)]
    end

    UI --> Chat
    MCP --> Chat
    HTTP --> Auth
    HTTP --> Docs
    HTTP --> Chat
    Chat --> Master
    Master --> Nested
    Nested --> Chroma
    Nested --> BM25
    Master --> PG
    Chat --> Redis
    Ops --> Prom[Prometheus]
    Prom --> Graf[Grafana]
```

| Layer | What it is |
| --- | --- |
| Product UI | Streamlit (`frontend/`). Local `run.py` uses port **8501**. Docker Compose publishes **8002** only. |
| API | FastAPI `main.py`. Local **8000**. Docker: internal `api:8000`, not published. |
| Chat | `POST /chat/` and `POST /chat` — SSE `token` then `done` (or `error`). |
| Orchestration | `app/agent/multi_agent/` master graph + nested `app/services/rag/` graph. |
| LLM | Anthropic Claude (`ClaudeService`), timeout + retry. Streaming on the chat path. |
| Dense search | Chroma cosine/HNSW, tenant filter `{user_id}`. |
| Lexical search | SQLite FTS5 BM25 in `data/bm25.db` (not the app DB). |
| Fusion | RRF (`k=60`) then `KnowledgeGraph` token co-occurrence boost then cross-encoder. |
| App state | Postgres via `DATABASE_URL` (`database/sqlite.py` name is historical). Users, documents, `user_memories`, LangGraph checkpoints. |
| Auth | JWT Bearer, secret required (≥32 chars). |
| Ingest | Sync pipeline, or Celery when `IS_ASYNC=true`. |

---

## 2. Layout (paths that actually exist)

```
basic_rag/
├── main.py                 FastAPI: routers, lifespan, /health, /metrics
├── run.py                  Local: uvicorn + Streamlit together
├── docker-compose.yml      redis, chroma, postgres, api, celery, ui, prometheus, grafana
├── track_production.py     Regenerates PRODUCTION_TRACKER.md
├── database/
│   ├── sqlite.py           Postgres engine + init_db (SQLAlchemy postgresql+psycopg)
│   └── models.py           User, Document, MemoryRecord
├── frontend/               Streamlit: login, ingest, chat SSE client
├── evals/                  Golden set, Ragas, DeepEval stand-in, MLflow log
├── observability/          Prometheus scrape + Grafana dashboard JSON
├── app/
│   ├── api/                HTTP: auth, documents, chat, session_access, dependencies
│   ├── agent/multi_agent/  Master graph, supervisor, specialists
│   ├── services/rag/       Nested CRAG graph
│   ├── services/retriever/ dense, bm25, hybrid, knowledge_graph, reranker
│   ├── services/llm/       Claude
│   ├── services/cache/     Redis answer cache
│   ├── guardrails/         Input/output rails, NeMo, Presidio, jailbreak
│   ├── memory/             Long-term extract + persist
│   ├── tools/              document_search, web_search, sql_*, invoke_tool
│   ├── mcp/server.py       MCP stdio: health, ask_rag
│   ├── core/               config, logger, audit, metrics, rate_limit, langfuse_tracer
│   └── tasks/              Celery ingest
```

---

## 3. How a chat turn runs (depth)

```mermaid
sequenceDiagram
    participant U as Streamlit / MCP / client
    participant API as FastAPI POST /chat/
    participant RL as SlowAPI
    participant Cache as Redis
    participant G as Master LangGraph
    participant RAG as Nested RAG graph
    participant LLM as Claude generate_stream
    participant Rails as Output guardrails
    participant Mem as Postgres memories

    U->>API: JWT + question + optional session_id
    API->>RL: 10/minute chat, 60/minute default
    API->>API: ensure_session_owner / new session_uuid
    API->>Cache: rag:answer:{user_id}:{sha256}
    alt cache hit
        Cache-->>U: SSE token + done
    else miss
        API->>G: ainvoke max_iterations=2
        G->>G: validate_input then load_memory
        G->>G: supervisor routes specialist
        opt rag_agent
            G->>RAG: ainvoke skip_generate thread_id session:rag
            RAG-->>G: context chunks
        end
        G->>G: synthesize then validate_output then save_memory
        G-->>API: trajectory + memories
        API->>LLM: stream tokens from context
        LLM-->>U: SSE type=token (many)
        API->>Rails: grounding + PII
        API->>Cache: SETEX
        API-->>U: SSE type=done
        API->>Mem: background persist_from_turn
    end
```

1. **Auth.** `Authorization: Bearer <jwt>`. Missing/invalid → 401. Rate limit exceeded → 429.
2. **Session.** Empty question → 400. If `session_id` exists and belongs to another user → 404. Else reuse or mint `session_<uuid>`.
3. **Keepalive.** First SSE line is `: keepalive` so the UI is not stuck while models load.
4. **Cache.** Redis key `rag:answer:{user_id}:{sha256(normalized question)}`. Hit skips the graphs (fail-open if Redis is down).
5. **Master graph** (`get_master_agent_graph`, compiled once). Checkpointer is `AsyncPostgresSaver` on `DATABASE_URL`. `thread_id` is the session id. Metadata includes `user_id` for ownership.
6. **Input rails.** Length cap 10k. Jailbreak denylist always. NeMo injection when enabled and ready. Presidio on input when spaCy loads.
7. **Long-term memory load.** Up to 10 `user_memories` rows for this `user_id`.
8. **Supervisor.** Cheap keyword router first; LLM `agenerate_structured(SupervisorDecision)` if ambiguous. After one specialist this turn, next hop is `FINISH`. Cap `max_iterations=2`.
9. **RAG specialist.** Nested graph with `skip_generate=True` and `skip_memory_persist=True`, `thread_id="{session}:rag"` so checkpoints do not collide. Retrieve is **pre-filtered** by `user_id`. Fallback tool: `document_search`.
10. **SQL / web / general.** Read-only SQL (denied tables include `users`, `user_memories`, checkpoints). Web: Tavily if `TAVILY_API_KEY` else DuckDuckGo. General: Claude + memories. Tools go through `invoke_tool` (audit + `requires_approval` for future write tools).
11. **Synthesize** then **output rails** on the supervisor answer; chat then **re-generates** with `generate_stream` so the UI sees tokens. Grounding = token overlap vs specialist context. Presidio may anonymize.
12. **SSE `done`.** `session_id`, question, answer, iterations, `agent_trajectory`, `guardrail_metadata`.
13. **Save memory.** Background thread extracts ADD/UPDATE/IGNORE (`semantic` / `episodic` allowed) into Postgres.
14. **Audit / metrics / traces.** `audit_event` (no raw PII). `inc_chat(ok|cache|blocked|error)`. LangSmith if tracing env is set. Langfuse if keys are set.

SSE event shapes:

| `type` | Meaning |
| --- | --- |
| `token` | Incremental Claude text (`content`) |
| `done` | Full payload; UI stores `session_id` |
| `error` | User-visible failure or safety block |

---

## 4. Master graph (only chat pipeline)

Compiled in `build_master_agent_graph`. Specialists always return to the supervisor so it can `FINISH`.

```mermaid
flowchart TD
    START([START]) --> validate_input
    validate_input -->|blocked| error_handler
    validate_input -->|continue| load_memory
    load_memory --> supervisor

    supervisor -->|rag_agent| rag_agent
    supervisor -->|sql_agent| sql_agent
    supervisor -->|web_agent| web_agent
    supervisor -->|general_agent| general_agent
    supervisor -->|FINISH| synthesize

    rag_agent --> supervisor
    sql_agent --> supervisor
    web_agent --> supervisor
    general_agent --> supervisor

    synthesize --> validate_output
    validate_output -->|continue| save_memory
    validate_output -->|blocked| error_handler
    save_memory --> END([END])
    error_handler --> END
```

| Node | Role |
| --- | --- |
| `validate_input` | Length, jailbreak, optional NeMo, optional Presidio |
| `load_memory` | Postgres long-term facts |
| `supervisor` | Route + `synthesize` |
| `rag_agent` | Nested hybrid CRAG |
| `sql_agent` | Schema → SELECT → summary |
| `web_agent` | Live search then grounded summary |
| `general_agent` | Direct Claude |
| `validate_output` | Grounding + PII on supervisor text |
| `save_memory` | Async extract/persist |
| `error_handler` | Safe refusal |

There is **no** public `/query/hybrid`. The nested RAG graph is only `rag_agent`.

---

## 5. Nested RAG graph (`rag_agent`)

`build_rag_graph` in `app/services/rag/graph.py`. Under the master, generate is skipped so FastAPI owns streaming.

```mermaid
flowchart TD
    START([START]) --> validate_input
    validate_input -->|blocked| error_handler
    validate_input -->|continue| load_memory
    load_memory --> retrieve
    retrieve -->|retry| retrieve
    retrieve -->|fallback| build_context
    retrieve -->|continue| reranker
    reranker --> grade_documents
    grade_documents -->|rewrite if empty retrieve| rewrite
    grade_documents -->|build_context| build_context
    rewrite --> retrieve
    build_context --> generate
    generate --> validate_output
    validate_output -->|retry| generate
    validate_output -->|continue| save_memory
    validate_output -->|error| error_handler
    save_memory --> END([END])
    error_handler --> END
```

Retrieve internals:

```mermaid
flowchart LR
    Q[Query + user_id filter] --> D[Dense Chroma]
    Q --> B[BM25 FTS5]
    D --> RRF[RRF k=60]
    B --> RRF
    RRF --> KG[KnowledgeGraph boost]
    KG --> CE[Cross-encoder top_k]
    CE --> CTX[Context builder]
```

Rewrite only runs when retrieve returned **no** documents and `retry_count < 1` (avoids a 5s rewrite on every weak hit). Rerank failures keep fused order.

---

## 6. Ingest graph (documents)

```mermaid
flowchart TD
    Upload[POST /ingest] --> Disk[documents/user_id/doc_id.ext]
    Disk --> Row[Postgres documents row]
    Row --> Sync{IS_ASYNC}
    Sync -->|false| Pipe[IngestionPipeline]
    Sync -->|true| Celery[Celery worker]
    Celery --> Pipe
    Pipe --> Load[Loader txt pdf md csv json]
    Load --> Clean[Cleaner]
    Clean --> Chunk[Recursive chunk 1000/200]
    Chunk --> Embed[Sentence transformer]
    Embed --> Chroma[Chroma upsert + user_id metadata]
    Chunk --> FTS[BM25Store data/bm25.db]
    Delete[DELETE /documents/id] --> ChromaDel[Delete vectors]
    Delete --> BM25Del[Delete FTS rows]
    Delete --> FileDel[Delete file + row]
```

---

## 7. Compose topology (Docker)

Public product port is **8002** (UI). API is only on the Docker network. Grafana/Prometheus bind **loopback** so they are not on the internet.

```mermaid
flowchart TB
    User((User)) -->|8002| UI[ui Streamlit]
    UI -->|http://api:8000| API[api uvicorn]
    API --> PG[(postgres)]
    API --> RD[(redis)]
    API --> CH[(chroma)]
    CW[celery_worker] --> PG
    CW --> RD
    CW --> CH
    Prom[prometheus :9090 loopback] -->|/metrics| API
    Graf[grafana :3000 loopback] --> Prom
```

---

## 8. All HTTP endpoints

OpenAPI: `http://localhost:8000/docs` (local) or via the API container internally. JWT: header `Authorization: Bearer <access_token>` except register/login/health/metrics.

### Ops (no JWT)

| Method | Path | What you get |
| --- | --- | --- |
| GET | `/health` | `{"status":"ok"}` for load balancers / Compose |
| GET | `/metrics` | Prometheus text: `rag_chat_requests_total{outcome=...}` |
| GET | `/docs` | Swagger UI |
| GET | `/redoc` | ReDoc |
| GET | `/openapi.json` | Schema |

### Auth (`app/api/auth.py`, prefix `/auth`)

| Method | Path | Auth | Body / notes |
| --- | --- | --- | --- |
| POST | `/auth/register` | no | JSON `{username, email, password}` → 201 `TokenResponse` |
| POST | `/auth/login` | no | **OAuth2 form**: `username` = **email**, `password` |
| GET | `/auth/me` | JWT | `UserInDB` |

### Chat (`app/api/query.py`, prefix `/chat`)

| Method | Path | Auth | Notes |
| --- | --- | --- | --- |
| POST | `/chat/` | JWT | Same handler as `/chat`. Body `ChatRequest`: `question`, optional `session_id`. SSE. Limit `RATE_LIMIT_CHAT` (default 10/minute). |
| POST | `/chat` | JWT | Identical |
| GET | `/chat/conversations` | JWT | Session ids this user owns (checkpoint metadata `user_id`) |
| GET | `/chat/conversations/{session_id}` | JWT | Messages from `PostgresSaver` (`chat_history` or `messages`) |

### Documents (`app/api/documents.py`, no prefix)

| Method | Path | Auth | Notes |
| --- | --- | --- | --- |
| POST | `/ingest` | JWT | Multipart file. Sync pipeline or Celery. |
| GET | `/tasks/{task_id}` | JWT | Celery status when async |
| GET | `/documents` | JWT | Current user’s rows |
| GET | `/documents/download/{document_id}` | JWT | Original file |
| DELETE | `/documents/{document_id}` | JWT | Chroma + BM25 + file + row |

### MCP (not HTTP)

`python -m app.mcp.server` — JSON-RPC stdio tools `health` and `ask_rag` (HTTP POST `/chat/` with `MCP_JWT` and `API_BASE_URL`).

---

## 9. Guardrails (what is wired)

`dependencies.py` builds both input and output services for the **master** graph and nested RAG.

| Rail | Wired? |
| --- | --- |
| Input length 10k | Yes |
| Jailbreak denylist | Yes (`JailbreakGuardrail`) |
| NeMo prompt injection | If `ENABLE_NEMO=true` and rails load; fail-closed on NeMo errors |
| Presidio PII in + out | If spaCy/Presidio import |
| Output grounding | Token overlap vs context (refusals score as grounded) |
| Output PII anonymize | If Presidio ready |
| Llama Guard | Only if you pass a provider (not constructed today) |
| Output JSON schema on chat | Off (free-form answers). Supervisor routing uses Pydantic structured output |

---

## 10. Memory

| Kind | Where | How to see it |
| --- | --- | --- |
| Short-term / conversation state | LangGraph checkpoints in Postgres, `thread_id` = session | `GET /chat/conversations/{id}`, LangSmith traces |
| Long-term | `user_memories` | Loaded each turn; extractor after `done` |
| Semantic / episodic | `memory_type` on those rows | Extractor JSON |
| Answer cache | Redis | Not conversation memory; same question replay |

---

## 11. How to run

### Prerequisites

- Python 3.11+ (Compose image is 3.13).
- `.env` with at least `JWT_SECRET_KEY` (32+ chars) and `ANTHROPIC_API_KEY` for answers.
- Postgres for app state. Local default `postgresql://rag:rag@localhost:5432/rag`. BM25 stays `data/bm25.db`.

### Local API + UI (`run.py`)

```bash
pip install -r requirements.txt
# Postgres must be up (Compose postgres service or local install)
python run.py
```

| URL | What |
| --- | --- |
| http://localhost:8000/docs | All endpoints |
| http://localhost:8000/health | Probe |
| http://localhost:8000/metrics | Counters |
| http://localhost:8501 | Streamlit |

Separate processes:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
streamlit run frontend/app.py
# Celery ingest:
# celery -A app.tasks.celery_app worker --loglevel=info
```

`python run.py --no-reload` matches production uvicorn (no `--reload`).

### Docker (production-shaped)

```bash
docker compose up --build
```

| URL | What |
| --- | --- |
| http://localhost:8002 | Streamlit (talks to `http://api:8000`) |
| http://127.0.0.1:9090 | Prometheus (host loopback) |
| http://127.0.0.1:3000 | Grafana (admin / `GRAFANA_ADMIN_PASSWORD` or `admin`) |

API is **not** published; use the UI or `docker compose exec api curl -s http://127.0.0.1:8000/health`.

### Evals

```bash
python evals/run_eval.py --offline
python evals/run_eval.py --with-ragas          # needs ANTHROPIC_API_KEY
python evals/run_eval.py --offline --with-deepeval
# MLflow: set MLFLOW_TRACKING_URI then run eval
python track_production.py                     # refresh PRODUCTION_TRACKER.md
```

### MCP (Cursor)

Command: `python -m app.mcp.server`. Env: `API_BASE_URL`, `MCP_JWT` (same JWT as `/auth/login`).

---

## 12. How to *see* logs, graphs, traces

### JSON application logs

`setup_logger()` in `main.py` writes:

- Console (JSON)
- `logs/rag.log` (rotating)
- `logs/error.log` (errors only)

Fields include `event`, `session_id`, `user_id`, `data` (audit: `chat.done`, `chat.blocked`, `tool.call`). Set `LOG_DIR` to change the folder. Docker: logs volume is the project `logs/` if the bind mount includes it (`.:/app`).

### LangGraph **structure** (this file)

The mermaid charts above **are** the graph. LangGraph does not host a public UI in this repo.

### LangGraph **runs** (LangSmith)

1. Set `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, optional `LANGSMITH_PROJECT`.
2. Restart API (`_configure_langsmith` sets `LANGCHAIN_TRACING_V2`).
3. Open [LangSmith](https://smith.langchain.com) → project → runs named `chat`. Nested RAG is a child run (`thread_id` `session_…:rag`).

### Prometheus

- Scrape: `GET /metrics`
- UI (Compose): http://127.0.0.1:9090 → Graph → query `rag_chat_requests_total`

### Grafana

- http://127.0.0.1:3000 (Compose)
- Provisioned datasource **Prometheus**, dashboard **RAG API** (`observability/grafana/dashboards/rag-api.json`)
- Panel: chat requests by `outcome` (`ok`, `cache`, `blocked`, `error`)

### Langfuse (optional)

Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, optional `LANGFUSE_HOST`. Chat outcomes are traced **without** prompt/answer PII. If the package or keys are missing, the API continues.

### Streamlit

Login → sidebar sessions + ingest → chat. Tokens appear as they stream. Session id comes from the `done` event.

---

## 13. Environment (`.env`)

Never commit secrets. Typical keys:

| Key | Role |
| --- | --- |
| `JWT_SECRET_KEY` | Required, ≥32 characters |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` | Claude |
| `DATABASE_URL` | Postgres (`postgresql://…`); SQLAlchemy uses `postgresql+psycopg://` |
| `REDIS_URL` | Cache + Celery |
| `CHROMA_HOST` / `CHROMA_PORT` / `CHROMA_PERSIST_DIRECTORY` | Vectors |
| `IS_ASYNC` | Celery ingest |
| `ENABLE_NEMO` / `NEMO_CONFIG_PATH` | Injection rails |
| `RATE_LIMIT_DEFAULT` / `RATE_LIMIT_CHAT` | SlowAPI |
| `RESPONSE_CACHE_TTL_SECONDS` | Redis TTL |
| `DENSE_TOP_K` / `BM25_TOP_K` / `RERANK_TOP_K` / `CHUNK_*` | Retrieval |
| `HF_TOKEN` | Hugging Face downloads |
| `TAVILY_API_KEY` | Preferred web search |
| `LANGSMITH_*` | Tracing |
| `LANGFUSE_*` | Optional traces |
| `MLFLOW_TRACKING_URI` | Eval experiment log |
| `API_BASE_URL` | Streamlit / MCP → API |
| `MCP_JWT` | MCP `ask_rag` |
| `GRAFANA_ADMIN_PASSWORD` | Compose Grafana |

Settings still lists unused MySQL fields; runtime identity is Postgres `users`.

---

## 14. Non-negotiables

- Chat = `POST /chat/` SSE `token` / `done` only.
- One supervisor + hybrid RAG specialist. No basic-RAG toggle.
- Sessions = `session_<uuid>`.
- Retrieval tenant filter happens **before** search, not after.
- Product UI in Docker is port **8002**; local `run.py` UI is **8501**.
