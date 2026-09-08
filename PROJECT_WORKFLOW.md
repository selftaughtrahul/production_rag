# RAG API — Project Overview

## Stack
| Layer | Technology |
|---|---|
| API | FastAPI |
| RAG Pipeline | LangGraph + Anthropic Claude |
| Vector Store | ChromaDB |
| Keyword Search | SQLite FTS5 (BM25) |
| Embeddings | HuggingFace Sentence Transformers |
| Memory / Sessions | LangGraph `SqliteSaver` → `rag_database.db` |
| Auth | JWT (Bearer token) |
| Observability | LangSmith |

---

## Project Structure

```
basic_rag/
├── main.py                          # App factory + router registration
├── database/
│   └── sqlite.py                   # DB connection, get_db(), init_db()
└── app/
    ├── api/
    │   ├── auth.py                  # POST /auth/register, /auth/login, GET /auth/me
    │   ├── documents.py             # POST /ingest, GET /documents, DELETE /documents/{id}
    │   ├── query.py                 # POST /query/, /query/hybrid, GET /query/conversations
    │   └── dependencies.py         # DI: graph builders, embedder, vector store
    ├── core/                        # Settings, logger, exceptions
    ├── models/
    │   └── schemas.py               # Pydantic request/response models
    ├── tasks/                       # Celery background tasks (async ingest)
    └── services/
        ├── auth/                    # JWT, password hashing, auth middleware
        ├── ingestion/               # document_loader, chunker, embedder, pipeline
        ├── llm/
        │   └── claude.py            # Anthropic Claude client
        ├── rag/                     # LangGraph: state, nodes, graph, edges
        ├── retriever/               # base, dense, bm25, hybrid, reranker, context
        └── vectorstore/             # ChromaDB wrapper
```

---

## Key Flows

### 1. Ingest a Document
```
POST /ingest
  → IngestionPipeline.ingest()
      → load → clean → chunk → embed
      → ChromaDB.add()  +  BM25Store.upsert()
      → save record in SQLite (documents table)
```

### 2. Query (Dense)
```
POST /query/
  → LangGraph (dense retriever)
      → rewrite query → retrieve (Chroma) → rerank → generate answer
      → checkpoint saved to rag_database.db (session memory)
```

### 3. Query (Hybrid)
```
POST /query/hybrid
  → LangGraph (hybrid retriever)
      → rewrite query
      → Dense (Chroma) + BM25 (SQLite FTS5)
      → RRF Fusion → rerank → generate answer
```

### 4. Conversation History
```
GET /query/conversations           → list all session IDs for current user
GET /query/conversations/{id}      → full message history for a session
```

---

## Retrieval Pipeline Detail

```
User Question
     │
     ▼
 Query Rewrite (Claude)
     │
     ├──────────────────────────────┐
     ▼                              ▼
Dense Retriever               BM25 Retriever
(ChromaDB cosine)             (SQLite FTS5)
     │                              │
     └──────────┬───────────────────┘
                ▼
         RRF Fusion (k=60)
                │
                ▼
        Cross-Encoder Reranker
                │
                ▼
         Claude LLM → Answer
```

---

## Environment Variables (`.env`)
| Key | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Claude API key |
| `LANGSMITH_API_KEY` | LangSmith tracing key |
| `EMBEDDING_MODEL` | HuggingFace model name |
| `CHROMA_PERSIST_DIR` | Path to ChromaDB storage |
| `JWT_SECRET` | JWT signing secret |

---

## Running Locally
```bash
# Start API
uvicorn main:app --reload

# Docs
http://localhost:8000/docs
```
