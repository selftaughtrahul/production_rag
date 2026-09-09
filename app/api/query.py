"""Query API — RAG invoke, streaming, and conversation history."""

import json
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from app.api.dependencies import get_hybrid_rag_graph, get_llm, get_rag_graph
from app.api.session_access import ensure_session_owner, list_owned_thread_ids
from app.memory.persist import persist_from_turn
from app.models.schemas import QueryRequest, UserInDB
from app.services.auth.dependencies import get_current_user
from app.services.llm.claude import ClaudeService

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/query", tags=["Query"])


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


def _build_graph_input(question: str, user_id: str, skip_generate: bool = False) -> dict:
    """Initial state passed into the RAG graph."""
    return {
        "question": question,
        "retry_count": 0,
        "user_id": user_id,
        "skip_generate": skip_generate,
    }


async def _run_retrieval_phase(rag_graph, question: str, user_id: str, thread_id: str) -> dict:
    """
    Run all graph nodes except the final LLM generation.

    Returns a dict with the keys we need to stream the answer:
        - context           (str)  retrieved + formatted document chunks
        - long_term_memories (list[str])  persisted facts about the user
        - chat_history      (list)  previous conversation turns
    """
    config = _build_graph_config(thread_id, user_id)
    return await rag_graph.ainvoke(
        _build_graph_input(question, user_id, skip_generate=True),
        config=config,
    )


async def _stream_answer(llm, question: str, state: dict) -> tuple[str, object]:
    """
    Async generator: stream the final answer token by token.

    This is a separate async generator so the endpoint can yield
    SSE lines as soon as each token arrives.

    Yields:
        str — individual text tokens

    Returns (via StopAsyncIteration value):
        tuple(full_answer, stream_object) — for post-stream processing
    """

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


async def _invoke_graph(rag_graph, question: str, user_id: str, session_id: str | None = None) -> dict:
    """Shared invocation logic for both query endpoints."""
    ensure_session_owner(session_id, user_id)
    thread_id = session_id or generate_new_session_id()

    result = await rag_graph.ainvoke(
        _build_graph_input(question, user_id),
        config=_build_graph_config(thread_id, user_id),
    )

    return {
        "success": True,
        "session_id": thread_id,
        "question": question,
        "answer": result.get("answer", ""),
    }





@router.post("/stream", summary="Query with token-by-token streaming (SSE)")
async def query_documents_stream(request: QueryRequest,rag_graph=Depends(get_rag_graph),llm: ClaudeService = Depends(get_llm),current_user: UserInDB = Depends(get_current_user)):
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

    ensure_session_owner(request.session_id, current_user.id)
    thread_id = request.session_id or generate_new_session_id()
    config = _build_graph_config(thread_id, current_user.id)

    async def event_generator():
        full_answer = ""

        try:
            state = await _run_retrieval_phase(
                rag_graph=rag_graph,
                question=request.question,
                user_id=current_user.id,
                thread_id=thread_id,
            )

            token_stream = await _stream_answer(
                llm=llm,
                question=request.question,
                state=state,
            )

            async for token in token_stream:
                if token:
                    full_answer += token
                    yield f'data: {json.dumps({"type": "token", "content": token})}\n\n'

            await rag_graph.aupdate_state(
                config,
                {
                    "question": request.question,
                    "answer": full_answer,
                    "chat_history": [
                        HumanMessage(content=request.question),
                        AIMessage(content=full_answer),
                    ],
                },
            )
            persist_from_turn(llm, current_user.id, request.question, full_answer)

            done_event = {
                "type": "done",
                "session_id": thread_id,
                "question": request.question,
                "answer": full_answer,
            }
            yield f"data: {json.dumps(done_event)}\n\n"

        except Exception as exc:
            logger.error("Streaming query failed: %s", exc, exc_info=True)
            error_event = {"type": "error", "error": "Query could not be completed."}
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
async def query_documents(request: QueryRequest,rag_graph=Depends(get_rag_graph),current_user: UserInDB = Depends(get_current_user)):
    """
    Ask a question using dense vector search only.

    Retrieval pipeline: Chroma Vector Search → Reranker → LLM
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    return await _invoke_graph(rag_graph, request.question, current_user.id, request.session_id)

@router.post("/hybrid", summary="Query (Hybrid: Dense + BM25 + RRF)")
async def query_documents_hybrid(request: QueryRequest,rag_graph=Depends(get_hybrid_rag_graph),current_user: UserInDB = Depends(get_current_user)):
    """
    Ask a question using hybrid retrieval (Dense Vector + BM25 Keyword).

    Retrieval pipeline:
        Dense Vector Search  ──┐
                               ├─→ RRF Fusion → Reranker → LLM
        BM25 Keyword Search  ──┘
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    return await _invoke_graph(rag_graph, request.question, current_user.id, request.session_id)

@router.get("/conversations", summary="List user conversations")
async def list_conversations(current_user: UserInDB = Depends(get_current_user)):
    """
    Fetch all unique session IDs associated with the current user
    by querying the langgraph checkpointer database.
    """
    return {"success": True, "sessions": list_owned_thread_ids(current_user.id)}

@router.get("/conversations/{session_id}",summary="Get conversation chat history")
async def get_conversation_history(session_id: str, current_user: UserInDB = Depends(get_current_user),):
    """
    Fetch the full chat history for a specific session from the SQLite checkpointer.
    """
    ensure_session_owner(session_id, current_user.id)

    config = {"configurable": {"thread_id": session_id}}

    with SqliteSaver.from_conn_string("rag_database.db") as cp:
        checkpoint = cp.get(config)

        if not checkpoint:
            raise HTTPException(status_code=404, detail="Conversation not found")

        raw_messages = checkpoint.get("channel_values", {}).get("chat_history", [])

        messages = []
        for m in raw_messages:
            if isinstance(m, dict):
                messages.append({"role": m.get("type", m.get("role", "unknown")), "content": m.get("content", "")})
            else:
                messages.append({"role": m.type, "content": m.content})

        return {
            "success": True,
            "session_id": session_id,
            "messages": messages,
        }

