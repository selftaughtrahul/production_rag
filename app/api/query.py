"""Single chat API — multi-agent orchestrator over hybrid retrieval."""

import json
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.sqlite import SqliteSaver

from app.api.dependencies import get_master_agent_graph
from app.api.session_access import ensure_session_owner, list_owned_thread_ids
from app.core.exceptions import LLMUnavailableError
from app.models.schemas import ChatRequest, UserInDB
from app.services.auth.dependencies import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])

CHAT_MODE = "agent"


def generate_new_session_id() -> str:
    return f"session_{uuid.uuid4()}"


def _build_graph_config(thread_id: str, user_id: str) -> dict:
    return {
        "configurable": {"thread_id": thread_id},
        "run_name": f"chat-{CHAT_MODE}",
        "tags": ["chat", CHAT_MODE],
        "metadata": {
            "source": "fastapi",
            "user_id": user_id,
            "session_id": thread_id,
            "mode": CHAT_MODE,
        },
    }


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


async def _run_agent(master_graph, question: str, user_id: str, thread_id: str) -> dict:
    result = await master_graph.ainvoke(
        {
            "query": question,
            "user_id": user_id,
            "thread_id": thread_id,
            "iterations": 0,
            "max_iterations": 2,
        },
        config=_build_graph_config(thread_id, user_id),
    )
    trajectory = [
        {
            "agent_name": out.agent_name,
            "result": out.result,
            "metadata": out.metadata,
        }
        for out in result.get("agent_outputs", [])
    ]
    return {
        "answer": result.get("final_response") or "Unable to synthesize answer.",
        "iterations": result.get("iterations", 1),
        "agent_trajectory": trajectory,
        "guardrail_metadata": result.get("guardrail_metadata", {}),
    }


@router.post("", summary="Chat (multi-agent orchestrator over hybrid retrieval)")
@router.post("/", summary="Chat (multi-agent orchestrator over hybrid retrieval)")
async def chat(
    request: ChatRequest,
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Single chat pipeline: the supervisor routes each turn to the RAG, SQL, web,
    or general specialist. The RAG specialist retrieves with dense + BM25 + RRF
    fusion and a cross-encoder rerank.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    ensure_session_owner(request.session_id, current_user.id)
    thread_id = request.session_id or generate_new_session_id()
    user_id = current_user.id

    async def event_generator():
        # Send headers immediately so the UI does not sit on a pending POST
        # while embeddings / the reranker / NeMo load on a cold start.
        yield ": keepalive\n\n"

        try:
            master_graph = await get_master_agent_graph()
            agent_result = await _run_agent(master_graph, question, user_id, thread_id)
            full_answer = agent_result["answer"]
            if full_answer:
                yield _sse({"type": "token", "content": full_answer})

            yield _sse(
                {
                    "type": "done",
                    "session_id": thread_id,
                    "question": question,
                    "answer": full_answer,
                    "mode": CHAT_MODE,
                    "iterations": agent_result["iterations"],
                    "agent_trajectory": agent_result["agent_trajectory"],
                    "guardrail_metadata": agent_result["guardrail_metadata"],
                }
            )
        except LLMUnavailableError as exc:
            # Expected provider condition (quota, key, outage) — not a code bug,
            # so report the real reason without a stack trace.
            logger.warning("Chat unavailable: %s", exc)
            yield _sse({"type": "error", "error": str(exc)})
        except Exception as exc:
            logger.error("Chat failed: %s", exc, exc_info=True)
            yield _sse({"type": "error", "error": "Query could not be completed."})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/conversations", summary="List user conversations")
async def list_conversations(current_user: UserInDB = Depends(get_current_user)):
    return {"success": True, "sessions": list_owned_thread_ids(current_user.id)}


@router.get("/conversations/{session_id}", summary="Get conversation chat history")
async def get_conversation_history(
    session_id: str,
    current_user: UserInDB = Depends(get_current_user),
):
    ensure_session_owner(session_id, current_user.id)

    config = {"configurable": {"thread_id": session_id}}

    with SqliteSaver.from_conn_string("rag_database.db") as cp:
        checkpoint = cp.get(config)

        if not checkpoint:
            raise HTTPException(status_code=404, detail="Conversation not found")

        raw_messages = checkpoint.get("channel_values", {}).get("chat_history", [])
        if not raw_messages:
            raw_messages = checkpoint.get("channel_values", {}).get("messages", [])

        messages = []
        for m in raw_messages:
            if isinstance(m, dict):
                messages.append(
                    {
                        "role": m.get("type", m.get("role", "unknown")),
                        "content": m.get("content", ""),
                    }
                )
            else:
                messages.append({"role": m.type, "content": m.content})

        return {
            "success": True,
            "session_id": session_id,
            "messages": messages,
        }
