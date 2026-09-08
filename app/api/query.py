from __future__ import annotations

import json
import sqlite3
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from app.api.dependencies import get_rag_graph, get_hybrid_rag_graph, get_llm
from app.models.schemas import QueryRequest, UserInDB
from app.observability import RAGObserver, RAGEvaluator
from app.services.auth.dependencies import get_current_user
from app.services.llm.claude import ClaudeService
from database.sqlite import get_connection

router = APIRouter(prefix="/query", tags=["Query"])


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def generate_new_session_id() -> str:
    """Generate a unique, readable session identifier."""
    return f"session_{uuid.uuid4()}"


def _build_graph_config(thread_id: str, user_id: str) -> dict:
    """
    Build the LangGraph invocation config.

    Centralised here so streaming and non-streaming paths always
    use the exact same config structure.
    """
    return {
        "configurable": {"thread_id": thread_id},
        "run_name": "rag-query",
        "tags": ["rag", "corrective-rag"],
        "metadata": {
            "source": "fastapi",
            "user_id": user_id,
            "session_id": thread_id,
        },
    }


def _build_graph_input(question: str, user_id: str) -> dict:
    """Initial state passed into the RAG graph."""
    return {
        "question": question,
        "retry_count": 0,
        "user_id": user_id,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Two-phase streaming
#
# Why two phases?
#
#   ClaudeService uses the raw Anthropic SDK (not a LangChain chat model),
#   so LangGraph's astream_events / on_chat_model_stream events never fire.
#
#   Instead we:
#     Phase 1 — Run the full graph synchronously (retrieve → rerank → grade
#               → build_context → load_memory → generate).  We intercept
#               the state *before* the final LLM call to get context + memories.
#     Phase 2 — Call ClaudeService.generate_stream() directly and yield
#               tokens to the client one by one.
#
#   This gives reliable, simple streaming without fighting the framework.
# ─────────────────────────────────────────────────────────────────────────────

async def _run_retrieval_phase(
    rag_graph,
    question: str,
    user_id: str,
    thread_id: str,
) -> dict:
    """
    Run all graph nodes except the final LLM generation.

    Returns a dict with the keys we need to stream the answer:
        - context           (str)  retrieved + formatted document chunks
        - long_term_memories (list[str])  persisted facts about the user
        - chat_history      (list)  previous conversation turns
    """
    # We run the full graph with generate() included so LangGraph handles
    # state persistence (checkpointer). The generate() node result is
    # thrown away — we re-run generation ourselves in streaming mode.
    config = _build_graph_config(thread_id, user_id)
    state = rag_graph.invoke(_build_graph_input(question, user_id), config=config)
    return state


async def _stream_answer(
    llm,
    question: str,
    state: dict,
) -> tuple[str, object]:
    """
    Async generator: stream the final answer token by token.

    This is a separate async generator so the endpoint can yield
    SSE lines as soon as each token arrives.

    Yields:
        str — individual text tokens

    Returns (via StopAsyncIteration value):
        tuple(full_answer, stream_object) — for post-stream processing
    """
    from langchain_core.messages import HumanMessage

    history_dicts = [
        {
            "role": "user" if isinstance(m, HumanMessage) else "assistant",
            "content": m.content,
        }
        for m in state.get("chat_history", [])
    ]

    return llm.generate_stream(
        question=question,
        context=state.get("context", ""),
        chat_history=history_dicts,
        long_term_memories=state.get("long_term_memories") or [],
    )

def _invoke_graph(rag_graph, question: str, user_id: str, session_id: str | None = None) -> dict:
    """Shared invocation logic for both query endpoints."""
    observer = RAGObserver()
    evaluator = RAGEvaluator()

    thread_id = session_id or generate_new_session_id()


    result = rag_graph.invoke(
        {
            "question": question,
            "retry_count": 0,
            "user_id": user_id,
        },
        config={
            "configurable": {"thread_id": thread_id},
            "run_name": question,
            "tags": ["rag", "corrective-rag"],
            "metadata": {"source": "fastapi", "user_id": user_id, "session_id": thread_id},
        },
    )

    metrics = observer.finish()
    evaluation = evaluator.evaluate(
        question=question,
        answer=result.get("answer", ""),
        context=result.get("context", ""),
    ).as_dict()

    return {
        "success": True,
        "session_id": thread_id,
        "question": question,
        "answer": result.get("answer", ""),
        "metrics": metrics,
        "evaluation": evaluation,
    }

@router.post(
    "/stream",
    summary="Query with token-by-token streaming (SSE)",
)
async def query_documents_stream(
    request: QueryRequest,
    rag_graph=Depends(get_rag_graph),
    llm: ClaudeService = Depends(get_llm),
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Ask a question and receive the answer streamed token by token.

    Response format: Server-Sent Events (SSE).
    Each line is:  data: <JSON>\n\n

    Event types:
        {"type": "token",   "content": "<text>"}     — one token
        {"type": "done",    "session_id": ..., "answer": ..., "metrics": ..., "evaluation": ...}
        {"type": "error",   "error": "<message>"}
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    thread_id = request.session_id or generate_new_session_id()
    observer = RAGObserver()
    evaluator = RAGEvaluator()

    async def event_generator():
        full_answer = ""

        try:
            # ── Phase 1: run full graph (retrieval + context building + memory) ──
            state = await _run_retrieval_phase(
                rag_graph=rag_graph,
                question=request.question,
                user_id=current_user.id,
                thread_id=thread_id,
            )

            # ── Phase 2: stream the answer token by token ─────────────────────
            token_stream = await _stream_answer(
                llm=llm,
                question=request.question,
                state=state,
            )

            async for token in token_stream:
                if token:
                    full_answer += token
                    yield f'data: {json.dumps({"type": "token", "content": token})}\n\n'

            # ── Phase 3: send final metadata ──────────────────────────────────
            metrics = observer.finish()
            evaluation = evaluator.evaluate(
                question=request.question,
                answer=full_answer,
                context=state.get("context", ""),
            ).as_dict()

            done_event = {
                "type": "done",
                "session_id": thread_id,
                "question": request.question,
                "answer": full_answer,
                "metrics": metrics,
                "evaluation": evaluation,
            }
            yield f"data: {json.dumps(done_event)}\n\n"

        except Exception as exc:
            error_event = {"type": "error", "error": str(exc)}
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",   # disable Nginx response buffering
        },
    )

@router.post("/", summary="Query (Dense retrieval)")
async def query_documents(
    request: QueryRequest,
    rag_graph=Depends(get_rag_graph),
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Ask a question using dense vector search only.

    Retrieval pipeline: Chroma Vector Search → Reranker → LLM
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    return _invoke_graph(rag_graph, request.question, current_user.id, request.session_id)


@router.post("/hybrid", summary="Query (Hybrid: Dense + BM25 + RRF)")
async def query_documents_hybrid(
    request: QueryRequest,
    rag_graph=Depends(get_hybrid_rag_graph),
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Ask a question using hybrid retrieval (Dense Vector + BM25 Keyword).

    Retrieval pipeline:
        Dense Vector Search  ──┐
                               ├─→ RRF Fusion → Reranker → LLM
        BM25 Keyword Search  ──┘
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    return _invoke_graph(rag_graph, request.question, current_user.id, request.session_id)


@router.get("/conversations", summary="List user conversations")
async def list_conversations(current_user: UserInDB = Depends(get_current_user)):
    """
    Fetch all unique session IDs associated with the current user
    by querying the langgraph checkpointer database.
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT thread_id, metadata FROM checkpoints GROUP BY thread_id")
        rows = cursor.fetchall()

        session_ids = set()
        for thread_id, metadata in rows:
            try:
                if metadata:
                    meta_dict = json.loads(metadata.decode("utf-8"))
                    if meta_dict.get("user_id") == current_user.id:
                        session_ids.add(thread_id)
            except Exception:
                continue

        return {"success": True, "sessions": list(session_ids)}
    except sqlite3.OperationalError:
        return {"success": True, "sessions": []}
    finally:
        conn.close()

@router.get(
    "/conversations/{session_id}",
    summary="Get conversation chat history",
)
async def get_conversation_history(
    session_id: str,
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Fetch the full chat history for a specific session from the SQLite checkpointer.
    """
    config = {"configurable": {"thread_id": session_id}}

    with SqliteSaver.from_conn_string("rag_database.db") as cp:
        checkpoint = cp.get(config)

        if not checkpoint:
            raise HTTPException(status_code=404, detail="Conversation not found")

        raw_messages = checkpoint.get("channel_values", {}).get("chat_history", [])

        messages = []
        for m in raw_messages:
            # LangGraph may deserialize messages as plain dicts or as message objects
            if isinstance(m, dict):
                messages.append({"role": m.get("type", m.get("role", "unknown")), "content": m.get("content", "")})
            else:
                messages.append({"role": m.type, "content": m.content})

        return {
            "success": True,
            "session_id": session_id,
            "messages": messages,
        }

