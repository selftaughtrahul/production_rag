"""Single chat API — basic RAG, hybrid RAG, or multi-agent via one endpoint."""

import json
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from app.api.dependencies import (
    get_hybrid_rag_graph,
    get_llm,
    get_master_agent_graph,
    get_rag_graph,
)
from app.api.session_access import ensure_session_owner, list_owned_thread_ids
from app.memory.persist import persist_from_turn_background
from app.models.schemas import ChatMode, ChatRequest, UserInDB
from app.services.auth.dependencies import get_current_user
from app.services.llm.claude import ClaudeService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


def generate_new_session_id() -> str:
    return f"session_{uuid.uuid4()}"


def _build_graph_config(thread_id: str, user_id: str, mode: ChatMode) -> dict:
    return {
        "configurable": {"thread_id": thread_id},
        "run_name": f"chat-{mode}",
        "tags": ["chat", mode],
        "metadata": {
            "source": "fastapi",
            "user_id": user_id,
            "session_id": thread_id,
            "mode": mode,
        },
    }


def _build_rag_input(question: str, user_id: str, skip_generate: bool = False) -> dict:
    return {
        "question": question,
        "retry_count": 0,
        "user_id": user_id,
        "skip_generate": skip_generate,
    }


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


async def _stream_rag_answer(llm: ClaudeService, question: str, state: dict):
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


async def _run_agent(master_graph, question: str, user_id: str, thread_id: str, mode: ChatMode) -> dict:
    result = await master_graph.ainvoke(
        {
            "query": question,
            "user_id": user_id,
            "thread_id": thread_id,
            "iterations": 0,
            "max_iterations": 2,
        },
        config=_build_graph_config(thread_id, user_id, mode),
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


@router.post("/", summary="Chat (basic RAG, hybrid RAG, or multi-agent)")
async def chat(
    request: ChatRequest,
    llm: ClaudeService = Depends(get_llm),
    current_user: UserInDB = Depends(get_current_user),
):
    """
    One chat endpoint. Choose the pipeline with `mode`:
    - basic: dense RAG subgraph
    - hybrid: dense + BM25 + RRF RAG
    - agent: supervisor multi-agent (RAG / SQL / web / general)
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    ensure_session_owner(request.session_id, current_user.id)
    thread_id = request.session_id or generate_new_session_id()
    mode = request.mode
    user_id = current_user.id
    config = _build_graph_config(thread_id, user_id, mode)

    async def event_generator():
        # Send headers immediately so the UI does not sit on a pending POST
        # while embeddings / the reranker / NeMo load on a cold start.
        yield ": keepalive\n\n"

        full_answer = ""
        extra = {
            "iterations": 0,
            "agent_trajectory": [],
            "guardrail_metadata": {},
        }

        try:
            if mode == "agent":
                master_graph = await get_master_agent_graph()
                agent_result = await _run_agent(
                    master_graph, question, user_id, thread_id, mode
                )
                full_answer = agent_result["answer"]
                extra = {
                    "iterations": agent_result["iterations"],
                    "agent_trajectory": agent_result["agent_trajectory"],
                    "guardrail_metadata": agent_result["guardrail_metadata"],
                }
                if full_answer:
                    yield _sse({"type": "token", "content": full_answer})
            else:
                graph = await (
                    get_hybrid_rag_graph() if mode == "hybrid" else get_rag_graph()
                )
                state = await graph.ainvoke(
                    _build_rag_input(question, user_id, skip_generate=True),
                    config=config,
                )
                token_stream = await _stream_rag_answer(llm, question, state)
                async for token in token_stream:
                    if token:
                        full_answer += token
                        yield _sse({"type": "token", "content": token})

                await graph.aupdate_state(
                    config,
                    {
                        "question": question,
                        "answer": full_answer,
                        "chat_history": [
                            HumanMessage(content=question),
                            AIMessage(content=full_answer),
                        ],
                    },
                )
                persist_from_turn_background(llm, user_id, question, full_answer)

            yield _sse(
                {
                    "type": "done",
                    "session_id": thread_id,
                    "question": question,
                    "answer": full_answer,
                    "mode": mode,
                    **extra,
                }
            )
        except Exception as exc:
            logger.error("Chat failed mode=%s: %s", mode, exc, exc_info=True)
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
