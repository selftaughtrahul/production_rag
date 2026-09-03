# RAG API — Complete Project Workflow

> **Purpose**: This single file gives any LLM full context about the project — architecture, data flows, all components, APIs, config, and conventions — so it can help without needing to browse the codebase.

---

> [!CAUTION]
> ## ALWAYS IGNORE — DO NOT READ
> **`venv/`** — The virtual environment folder contains thousands of third-party library files.
> **Never read, list, search, or include any file inside `venv/`** when working on this project.
> Doing so wastes tokens and provides zero value. Treat `venv/` as if it does not exist.
> This applies to all LLMs, agents, and tools at all times without exception.

---

## 1. Project Overview

A **production-oriented Retrieval-Augmented Generation (RAG) API** built with:

| Layer | Technology |
|---|---|
| API Framework | FastAPI |
| LLM | Anthropic Claude (`claude-sonnet-4-5`) |
| Embeddings | HuggingFace Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector Store | ChromaDB (local persistent or HTTP client) |
| Reranker | Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| RAG Orchestration | LangGraph (StateGraph) |
| Background Tasks | Celery + Redis |
| Observability | LangSmith tracing + custom RAGObserver metrics |
| Containerization | Docker + Docker Compose |

**Running mode**: Dev uses synchronous ingestion via FastAPI. Production (Linux) uses Celery async tasks.

---

## 2. Repository Structure

```
basic_rag/
├── main.py                          # FastAPI app entry point — all HTTP routes
├── requirements.txt                 # All pinned Python dependencies
├── .env / .env.example              # Environment variables
├── Dockerfile                       # Container build
├── docker-compose.yml               # Orchestrates: app + chromadb + redis
├── documents/temp/                  # Temp storage for uploaded files
├── data/chroma_db/                  # ChromaDB local persistence (default)
└── app/
    ├── core/
    │   ├── config.py                # Settings dataclass — reads from .env
    │   ├── logger.py                # Logging setup
    │   └── exceptions.py            # Custom exceptions
    ├── api/
    │   ├── dependencies.py          # DI: builds all components (cached)
    │   └── query.py                 # POST /query/ endpoint + router
    ├── pipeline/
    │   └── pipeline.py              # IngestionPipeline — orchestrates ingestion steps
    ├── services/
    │   ├── ingestion/
    │   │   ├── documnent_loader.py  # DocumentLoaderLibrary (LlamaIndex-based)
    │   │   ├── data_cleaning.py     # DataCleaningCustom / DataCleaningLibrary (ftfy + clean-text)
    │   │   ├── text_normalizer.py   # Text normalization utilities
    │   │   ├── chunker.py           # Chunking strategies + ChunkerService
    │   │   └── embedding.py         # HuggingFaceEmbeddingProvider + EmbeddingService
    │   ├── vectorstore/
    │   │   ├── base.py              # Abstract VectorStore, EmbeddedChunk, SearchResult
    │   │   ├── chroma.py            # ChromaVectorStore implementation
    │   │   └── models.py            # Pydantic models for vector store
    │   ├── retriever/
    │   │   ├── retriever.py         # Abstract Retriever interface
    │   │   ├── service.py           # RetrieverService (concrete implementation)
    │   │   ├── reranker.py          # CrossEncoderReranker
    │   │   ├── context.py           # ContextBuilder (formats docs into a string)
    │   │   ├── metadata_filter.py   # Metadata filter helpers
    │   │   └── hybrid_search.py     # (stub) Hybrid search
    │   ├── rag/
    │   │   ├── state.py             # RAGState TypedDict (LangGraph state)
    │   │   ├── graph.py             # build_rag_graph() — wires the LangGraph
    │   │   ├── nodes.py             # RAGNodes class — retrieve, grade, rewrite, generate
    │   │   └── edge.py              # decide_after_grading() conditional edge
    │   └── llm/
    │       └── claude.py            # ClaudeService — generate(), rewrite_query()
    ├── observability/
    │   ├── __init__.py              # Exports RAGObserver, RAGEvaluator
    │   ├── metrics.py               # RAGObserver — timing + event tracking
    │   ├── evaluation.py            # RAGEvaluator — basic answer quality evaluation
    │   └── callbacks.py             # LangSmith callback integration
    ├── tasks/
    │   ├── celery_app.py            # Celery application instance
    │   └── tasks.py                 # ingest_document_task (Celery task)
    ├── models/                      # (Pydantic request/response models)
    └── query/                       # (query-level helpers)
```

---

## 3. Environment Configuration

All settings live in `.env`. Loaded by `app/core/config.py` via `Settings.from_environment()`.

```env
# Redis (for Celery background tasks)
REDIS_URL=redis://localhost:6379/0

# ChromaDB — leave CHROMA_HOST empty -> uses local PersistentClient
CHROMA_HOST=
CHROMA_PORT=8001
CHROMA_PERSIST_DIRECTORY=data/chroma_db

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=          # e.g. "cuda" or leave blank for CPU
RAG_COLLECTION_PREFIX=rag
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Reranker
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# LLM (Anthropic Claude)
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-5

# LangSmith observability (optional)
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=default
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

The `Settings` class is a **frozen dataclass** — immutable at runtime. `build_components()` and `get_rag_graph()` are `@lru_cache(maxsize=1)` — built once per process.

---

## 4. API Endpoints

### Base URL: `http://localhost:8000`

| Method | Path | Description |
|---|---|---|
| `POST` | `/ingest` | Upload a document file; triggers ingestion pipeline |
| `GET` | `/tasks/{task_id}` | Poll a Celery background ingestion task |
| `DELETE` | `/documents/{document_id}` | Delete a document and all its vector chunks |
| `POST` | `/query/` | Ask a question; returns answer + metrics + evaluation |

---

### POST `/ingest`

**Request**: `multipart/form-data` with a `file` field.

**Process** (synchronous dev mode):
1. Validate file is not empty.
2. Generate a `document_id` (UUID4).
3. Save file to `documents/temp/` as a named temp file.
4. Build `IngestionPipeline` and call `pipeline.ingest()`.
5. Return result.

**Response**:
```json
{
  "success": true,
  "message": "Document upload successful",
  "document_id": "uuid-string",
  "filename": "myfile.pdf",
  "ingestion": {
    "document_id": "uuid-string",
    "filename": "myfile.pdf",
    "chunks_count": 42
  }
}
```

---

### GET `/tasks/{task_id}`

**Request**: `task_id` from a Celery task (Linux async mode only).

**Response**:
```json
{
  "task_id": "celery-task-id",
  "status": "SUCCESS",
  "result": {}
}
```

---

### DELETE `/documents/{document_id}`

**Process**:
1. Count chunks for the `document_id` in ChromaDB.
2. If 0 chunks -> 404.
3. Delete all chunks with `vector_store.delete_document(document_id)`.

**Response**:
```json
{
  "success": true,
  "message": "Document deleted successfully",
  "document_id": "uuid-string",
  "deleted_chunks": 42
}
```

---

### POST `/query/`

**Request body**:
```json
{ "question": "What is the refund policy?" }
```

**Process**: Invokes the LangGraph RAG graph (see section 7).

**Response**:
```json
{
  "success": true,
  "question": "What is the refund policy?",
  "answer": "The refund policy states...",
  "metrics": { "retrieval_time": 0.12, "generation_time": 1.4 },
  "evaluation": { "relevance": 0.85 }
}
```

---

## 5. Ingestion Pipeline — Step by Step

**Entry**: `IngestionPipeline.ingest(source, document_id, filename)`
**File**: `app/pipeline/pipeline.py`

```
File Upload
    |
    v
[1] DocumentLoaderLibrary.load(source)
    -> Reads PDF / DOCX / TXT / etc. via LlamaIndex
    -> Returns Document(text=..., metadata={...})
    |
    v
[2] DataCleaningLibrary.clean(document.text)
    -> ftfy: fix broken Unicode / mojibake
    -> clean-text: remove URLs, emails, emojis, HTML
    -> Normalize whitespace
    -> Returns cleaned plain text string
    |
    v
[3] ChunkerService.chunk(cleaned_text, metadata=document_metadata)
    -> Strategy: LangChainRecursiveStrategy (wraps LangChain splitter)
    -> chunk_size=1000 chars, chunk_overlap=200 chars
    -> Each chunk enriched with metadata:
        { document_id, filename, chunk_index, chunk_size }
    -> Returns list[Chunk(text, index, metadata)]
    |
    v
[4] EmbeddingService.embed_chunks(chunks)
    -> HuggingFace model: sentence-transformers/all-MiniLM-L6-v2
    -> Embedding dimension: 384
    -> Returns list[EmbeddedChunk(chunk_id, text, embedding, metadata)]
    -> chunk_id format: "{document_id}_chunk_{index}"
    |
    v
[5] ChromaVectorStore.add(embedded_chunks)
    -> Upserts to ChromaDB collection (cosine similarity space)
    -> Collection name: "rag_{model_slug}_{dimension}"
    |
    v
Returns: { document_id, filename, chunks_count }
```

---

## 6. Chunking Strategies

**File**: `app/services/ingestion/chunker.py`

Three strategies are implemented (all inherit from `ChunkingStrategy` ABC):

| Class | Description |
|---|---|
| `FixedSizeChunkingStrategy` | Simple character-based sliding window |
| `RecursiveChunkingStrategy` | Custom recursive split: paragraph -> line -> sentence -> word -> char |
| `LangChainRecursiveStrategy` | Adapter wrapping LangChain RecursiveCharacterTextSplitter (ACTIVE) |

**Active strategy** (wired in `dependencies.py`): `LangChainRecursiveStrategy`

`ChunkerService` wraps the strategy and adds index + size metadata to each `Chunk`.

---

## 7. RAG Query Flow — LangGraph

**File**: `app/services/rag/graph.py`, `nodes.py`, `edge.py`, `state.py`

This is a **Corrective RAG** pattern using LangGraph's `StateGraph`.

### State: `RAGState` (TypedDict)
```python
{
  "question": str,
  "rewritten_question": str | None,
  "documents": list[SearchResult],
  "documents_relevant": bool,
  "context": str,
  "answer": str,
  "retry_count": int,
  "observer": RAGObserver
}
```

### Graph Topology

```
START
  |
  v
[retrieve]
  -> embed question (or rewritten_question)
  -> search ChromaDB top_k=10
  -> CrossEncoder rerank -> top_k=5
  |
  v
[grade_documents]
  -> Filter docs with rerank_score >= -1.0
  -> If relevant docs exist -> documents_relevant=True
  -> If no relevant docs   -> documents_relevant=False
  |
  |--- documents_relevant=True ---> [build_context]
  |                                       |
  |                                       v
  |                                  [generate]
  |                                       |
  |                                       v
  |                                      END
  |
  |--- documents_relevant=False --> [rewrite]
                                        |
                                        v
                                   [retrieve]  (loops back)
```

### Node Details

| Node | Class method | What it does |
|---|---|---|
| `retrieve` | `RAGNodes.retrieve` | Embeds query -> vector search (top 10) -> rerank (top 5) |
| `grade_documents` | `RAGNodes.grade_documents` | Checks rerank scores; decides relevant or not |
| `rewrite` | `RAGNodes.rewrite_query` | Calls `ClaudeService.rewrite_query()` to rephrase the question |
| `build_context` | `RAGNodes.build_context` | `ContextBuilder.build(docs)` -> formatted string |
| `generate` | `RAGNodes.generate` | Calls `ClaudeService.generate(question, context)` -> final answer |

### Conditional Edge

`decide_after_grading(state)` in `edge.py`:
- Returns `"generate"` if `state["documents_relevant"] == True`
- Returns `"rewrite"` if `state["documents_relevant"] == False`

---

## 8. Embedding Service

**File**: `app/services/ingestion/embedding.py`

```
HuggingFaceEmbeddingProvider
  ├── model: sentence-transformers/all-MiniLM-L6-v2
  ├── dimension: 384
  ├── embed_query(text: str) -> list[float]          # single query
  └── embed_texts(texts: list[str]) -> list[list[float]]  # batch

EmbeddingService
  └── embed_chunks(chunks: list[Chunk]) -> list[EmbeddedChunk]
      # Assigns chunk_id = "{document_id}_chunk_{index}"
```

Model is loaded **once** via `@lru_cache` in `dependencies.py`.

---

## 9. Vector Store — ChromaDB

**File**: `app/services/vectorstore/chroma.py`

| Mode | Config |
|---|---|
| Local (dev) | `PersistentClient(path="data/chroma_db")` — CHROMA_HOST is empty |
| Remote (Docker) | `HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)` |

**Collection naming**: `rag_{model_slug}_{embedding_dimension}`
Example: `rag_sentence_transformers_all_minilm_l6_v2_384`

**Similarity metric**: Cosine (`hnsw:space: cosine`)

**Key methods**:
- `add(chunks)` — upsert vectors
- `search(query_embedding, top_k, metadata_filter)` — cosine search
- `delete_document(document_id)` — deletes all chunks by `document_id` metadata filter
- `count_document_chunks(document_id)` — counts chunks for a document
- `document_exists(document_id)` — bool check

**Score conversion**: ChromaDB returns cosine distance -> converted to score: `score = 1.0 / (1.0 + distance)`

---

## 10. LLM — Claude Service

**File**: `app/services/llm/claude.py`

```
ClaudeService
  ├── model: "claude-sonnet-4-5" (default, overridable via ANTHROPIC_MODEL env)
  ├── generate(question, context) -> str
  |     # Final RAG answer — strict system prompt: answer ONLY from context
  ├── rewrite_query(question) -> str
  |     # Rewrites user question into a concise semantic search query (<=12 words)
  └── generate_text(prompt, system_prompt, max_tokens) -> str
        # Generic Claude call for any LLM task
```

**System prompt for generate()**: Instructs Claude to answer strictly from context, no hallucination, no filler phrases.

---

## 11. Reranker

**File**: `app/services/retriever/reranker.py`

```
CrossEncoderReranker
  └── model: cross-encoder/ms-marco-MiniLM-L-6-v2
  └── rerank(query, documents, top_k=5)
        -> pairs = [(query, doc.text) for each doc]
        -> scores = CrossEncoder.predict(pairs)
        -> Sort by score descending
        -> Attach rerank_score to doc.metadata
        -> Return top_k docs
```

Model loaded once via `@lru_cache`.

---

## 12. Data Cleaning

**File**: `app/services/ingestion/data_cleaning.py`

Two implementations (both available):

### `DataCleaningLibrary` (ACTIVE — used in `dependencies.py`)
Library-backed cleaner using `ftfy` + `clean-text`:
1. `ftfy.fix_text()` — fix broken Unicode/mojibake
2. `clean()` — remove URLs, emails, emojis, HTML (configurable)
3. Normalize whitespace (preserve meaningful line breaks)

### `DataCleaningCustom`
Pure custom implementation without third-party libs:
1. Normalize Unicode (NFKC)
2. Remove HTML tags/entities
3. Remove URLs, emails
4. Remove emojis (Unicode ranges)
5. Remove control characters
6. Reduce repeated characters (`heyyyy` -> `heyy`)
7. Normalize spaces

---

## 13. Document Loader

**File**: `app/services/ingestion/documnent_loader.py`

`DocumentLoaderLibrary` uses **LlamaIndex** (`llama-index-readers-file`) to load:
- PDF (`.pdf`) via `pypdf`
- DOCX, TXT, Markdown, and other formats supported by LlamaIndex

Returns a `Document(text: str, metadata: dict)` object.

---

## 14. Observability

**Files**: `app/observability/`

### `RAGObserver` (`metrics.py`)
Tracks timing and events during a RAG query:
- `on_retrieval_start()` / `on_retrieval_end(start_time, document_count)`
- `on_generation_start()` / `on_generation_end(start_time)`
- `on_documents_graded(count)`
- `on_query_rewritten()`
- `on_retry()`
- `finish()` -> returns metrics dict

### `RAGEvaluator` (`evaluation.py`)
Basic quality evaluation:
- `evaluate(question, answer, context)` -> `EvaluationResult`
- `.as_dict()` -> dict with evaluation scores

### LangSmith Tracing (`callbacks.py`)
Optional — enabled via `LANGSMITH_TRACING=true` in `.env`. Traces the full LangGraph execution.

---

## 15. Background Tasks (Celery)

**Files**: `app/tasks/`

```python
# celery_app.py
celery_app = Celery("rag_tasks", broker=REDIS_URL, backend=REDIS_URL)

# tasks.py
@celery_app.task
def ingest_document_task(source, document_id, filename):
    pipeline = build_ingestion_pipeline(source=source)
    return pipeline.ingest(source, document_id, filename)
```

**Note**: Currently only used on Linux (`is_linux = True` in `main.py`). On Windows/dev, ingestion runs synchronously in the request thread.

---

## 16. Dependency Injection Pattern

**File**: `app/api/dependencies.py`

All shared components are built once and cached:

```python
@lru_cache(maxsize=1)
def build_components() -> RAGComponents:
    # Creates: EmbeddingService + ChromaVectorStore

@lru_cache(maxsize=1)
def get_rag_graph() -> compiled LangGraph:
    # Creates: RetrieverService + CrossEncoderReranker + ContextBuilder + ClaudeService
    # Wires and compiles the LangGraph

def build_ingestion_pipeline(source) -> IngestionPipeline:
    # Creates a fresh IngestionPipeline per ingestion request
    # Reuses cached EmbeddingService + ChromaVectorStore
```

---

## 17. Docker / Deployment

**`docker-compose.yml`** starts three services:
1. **`app`** — FastAPI + Uvicorn (this application)
2. **`chromadb`** — ChromaDB HTTP server
3. **`redis`** — Redis broker for Celery

In Docker mode, `CHROMA_HOST=chroma` is set, so the app connects to the ChromaDB container via HTTP instead of local persistence.

**Run locally (dev)**:
```bash
uvicorn main:app --reload
# App runs at http://localhost:8000
# Swagger UI at http://localhost:8000/docs
```

**Run with Docker**:
```bash
docker-compose up --build
```

---

## 18. Key Design Decisions & Conventions

| Decision | Choice | Reason |
|---|---|---|
| Chunking strategy | `LangChainRecursiveStrategy` | Battle-tested, respects paragraph/sentence boundaries |
| Embedding | HuggingFace `all-MiniLM-L6-v2` | Fast, local, free, good quality (384-dim) |
| Vector DB | ChromaDB | Simple, runs locally without infra, supports cosine |
| LLM | Anthropic Claude | High quality, strict instruction following |
| Reranker | Cross-Encoder | Significantly improves retrieval precision over bi-encoder alone |
| RAG pattern | Corrective RAG (CRAG) | Loops query rewriting if retrieved docs are irrelevant |
| State management | LangGraph `StateGraph` | Clean node/edge separation, easy to extend |
| DI / caching | `@lru_cache` | Models are expensive to load; cache for process lifetime |
| Dev mode | Synchronous ingestion | Avoids Celery/Redis setup overhead on Windows |
| Metadata | `document_id` in every chunk | Enables per-document operations (delete, filter) |

---

## 19. Data Flow Summary

### Ingestion
```
User uploads file
  -> save to temp dir
  -> load with LlamaIndex
  -> clean text (ftfy + clean-text)
  -> chunk (LangChain recursive, 1000 chars / 200 overlap)
  -> embed each chunk (HuggingFace, 384-dim)
  -> upsert to ChromaDB (cosine space)
```

### Query
```
User sends question
  -> embed question (same HF model)
  -> vector search ChromaDB top-10
  -> CrossEncoder rerank -> top-5
  -> grade relevance via rerank scores
  -> [if not relevant] -> Claude rewrites query -> retry search
  -> [if relevant] -> build context string
  -> Claude generates answer strictly from context
  -> return answer + timing metrics + quality evaluation
```

---

## 20. File-by-File Quick Reference

| File | Purpose |
|---|---|
| `main.py` | FastAPI app, 3 routes: /ingest, /tasks/{id}, /documents/{id} |
| `app/core/config.py` | Settings frozen dataclass — all env var reading |
| `app/api/dependencies.py` | DI factory functions, all @lru_cache singletons |
| `app/api/query.py` | POST /query/ endpoint, invokes LangGraph, returns answer + metrics |
| `app/pipeline/pipeline.py` | IngestionPipeline — orchestrates steps 1-5 of ingestion |
| `app/services/ingestion/documnent_loader.py` | LlamaIndex-based file reader |
| `app/services/ingestion/data_cleaning.py` | DataCleaningLibrary (active) + DataCleaningCustom |
| `app/services/ingestion/chunker.py` | ChunkerService + 3 strategies; active = LangChainRecursiveStrategy |
| `app/services/ingestion/embedding.py` | HuggingFace embedder, EmbeddingService |
| `app/services/vectorstore/chroma.py` | ChromaVectorStore — add/search/delete/count |
| `app/services/retriever/service.py` | RetrieverService — embed query -> search ChromaDB |
| `app/services/retriever/reranker.py` | CrossEncoderReranker — re-scores docs with cross-encoder |
| `app/services/retriever/context.py` | ContextBuilder — formats retrieved docs into a single string |
| `app/services/rag/graph.py` | build_rag_graph() — wires LangGraph nodes and edges |
| `app/services/rag/nodes.py` | RAGNodes — retrieve, grade, rewrite, build_context, generate |
| `app/services/rag/edge.py` | decide_after_grading() — conditional routing logic |
| `app/services/rag/state.py` | RAGState TypedDict — shared state across graph nodes |
| `app/services/llm/claude.py` | ClaudeService — generate answer + rewrite query |
| `app/observability/metrics.py` | RAGObserver — per-request timing + event counters |
| `app/observability/evaluation.py` | RAGEvaluator — basic answer quality evaluation |
| `app/tasks/celery_app.py` | Celery app instance |
| `app/tasks/tasks.py` | ingest_document_task Celery task |
