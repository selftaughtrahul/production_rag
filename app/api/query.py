"""Single chat API — multi-agent orchestrator over hybrid retrieval."""

import asyncio
import json
import logging
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from app.api.dependencies import (
    build_wired_output_guardrails,
    get_checkpointer,
    get_llm,
    get_master_agent_graph,
)
from app.api.session_access import (
    ensure_session_owner,
    list_owned_conversations,
    thread_owner,
)
from app.core.audit import audit_event
from app.core.exceptions import LLMUnavailableError
from app.core.metrics import inc_chat
from app.core.rate_limit import CHAT_LIMIT, limiter
from app.models.schemas import ChatRequest, UserInDB
from app.prompts.specialist_prompts import RAG_NO_EVIDENCE_ANSWER
from app.services.auth.dependencies import get_current_user
from app.services.cache.response_cache import get_cached_answer, set_cached_answer
from .helper import (
    _iter_display_sse,
    _sse,
    _build_graph_config,
    generate_new_session_id,
    _run_agent,
    _context_from_agent,
)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])






@router.post("", summary="Chat (multi-agent orchestrator over hybrid retrieval)")
@router.post("/", summary="Chat (multi-agent orchestrator over hybrid retrieval)")
@limiter.limit(CHAT_LIMIT)
async def chat(request: Request,body: ChatRequest,current_user: UserInDB = Depends(get_current_user)):
    """
    Single chat pipeline: the supervisor routes each turn to the RAG, web,
    or general specialist. The RAG specialist retrieves with dense + BM25 + RRF
    fusion and a cross-encoder rerank.
    """
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    ensure_session_owner(body.session_id, current_user.id)
    thread_id = body.session_id or generate_new_session_id()
    user_id = current_user.id

    async def event_generator():
        # Send headers immediately so the UI does not sit on a pending POST
        # while embeddings / the reranker / NeMo load on a cold start.
        yield ": keepalive\n\n"

        try:
            # Redis response cache: key = user_id + query hash.
            # Check before the graph so a hit does not load models.
            cached_answer = get_cached_answer(user_id, question)
            if cached_answer:
                logger.info("response_cache hit user_id=%s", user_id)
                inc_chat("cache")
                audit_event("chat.done",user_id=user_id,session_id=thread_id,data={"outcome": "cache"})
                async for frame in _iter_display_sse(cached_answer):
                    yield frame
                
                yield _sse(
                    {
                        "type": "done",
                        "session_id": thread_id,
                        "question": question,
                        "answer": cached_answer,
                        "iterations": 0,
                        "agent_trajectory": [],
                        "guardrail_metadata": {"response_cache": "hit"},
                        "sources": [],
                    }
                )
                return

            master_graph = await get_master_agent_graph()
            agent_result = await _run_agent(master_graph, question, user_id, thread_id)
            context = _context_from_agent(agent_result)
            print("is direct response",agent_result.get("direct_response"))
            llm = get_llm()
            pieces: list[str] = []
            streamed = False
            if agent_result.get("direct_response"):
                pieces.append(str(agent_result["answer"]))
            else:
                async for token in llm.generate_stream(question=question,context=context,long_term_memories=agent_result.get("long_term_memories") or [],):
                    if not token:
                        continue
                    pieces.append(token)
                    streamed = True
                    yield _sse({"type": "token", "content": token})

            full_answer = "".join(pieces).strip() or agent_result["answer"]
            rails = build_wired_output_guardrails()
            rail_result = await rails.validate(
                query=question,
                response=full_answer,
                context=context,
            )
            guardrail_metadata = {
                **(agent_result.get("guardrail_metadata") or {}),
                **(rail_result.metadata or {}),
            }
            if not rail_result.passed and rail_result.action == "block":
                safe_answer = "The answer was blocked by a safety check."
                inc_chat("blocked")
                audit_event(
                    "chat.blocked",
                    user_id=user_id,
                    session_id=thread_id,
                    data={"stage": "output_guardrail"},
                )
                if not streamed:
                    async for frame in _iter_display_sse(safe_answer):
                        yield frame
                yield _sse(
                    {
                        "type": "done",
                        "session_id": thread_id,
                        "question": question,
                        "answer": safe_answer,
                        "iterations": agent_result["iterations"],
                        "agent_trajectory": agent_result["agent_trajectory"],
                        "guardrail_metadata": guardrail_metadata,
                        "sources": [],
                    }
                )
                return
            if rail_result.metadata.get("anonymized_response"):
                full_answer = rail_result.metadata["anonymized_response"]
            if not streamed:
                async for frame in _iter_display_sse(full_answer):
                    yield frame

            # Do not cache a refusal: the user may upload the missing document next.
            if (
                agent_result.get("input_guardrail_passed", True)
                and full_answer != RAG_NO_EVIDENCE_ANSWER
            ):
                set_cached_answer(user_id, question, full_answer)
            inc_chat("ok")
            audit_event("chat.done",user_id=user_id,session_id=thread_id,data={"outcome": "ok","agents": [item.get("agent_name") for item in (agent_result.get("agent_trajectory") or [])]})

            yield _sse(
                {
                    "type": "done",
                    "session_id": thread_id,
                    "question": question,
                    "answer": full_answer,
                    "iterations": agent_result["iterations"],
                    "agent_trajectory": agent_result["agent_trajectory"],
                    "guardrail_metadata": guardrail_metadata,
                    "sources": agent_result.get("sources") or [],
                }
            )
        except LLMUnavailableError as exc:

            logger.warning("Chat unavailable: %s", exc)
            inc_chat("error")
            audit_event("chat.error",user_id=user_id,session_id=thread_id,data={"outcome": "llm_unavailable"})
            yield _sse({"type": "error", "error": str(exc)})
        except Exception as exc:
            logger.error("Chat failed: %s", exc, exc_info=True)
            inc_chat("error")
            audit_event("chat.error",user_id=user_id,session_id=thread_id,data={"outcome": "failed"})
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
    conversations = list_owned_conversations(current_user.id)
    return {"success": True, "sessions": conversations}


@router.delete("/conversations/{session_id}", summary="Delete a conversation thread")
async def delete_conversation(session_id: str,current_user: UserInDB = Depends(get_current_user)):
    """Remove one owned chat thread and its nested RAG checkpoint."""
    if not session_id.startswith("session_") or ":" in session_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if thread_owner(session_id) != current_user.id:
        raise HTTPException(status_code=404, detail="Conversation not found")

    checkpointer = await get_checkpointer()
    await checkpointer.adelete_thread(session_id)
    await checkpointer.adelete_thread(f"{session_id}:rag")
    return {"success": True, "session_id": session_id}


@router.get("/conversations/{session_id}", summary="Get conversation chat history")
async def get_conversation_history(session_id: str,current_user: UserInDB = Depends(get_current_user),):
    ensure_session_owner(session_id, current_user.id)

    config = {"configurable": {"thread_id": session_id}}

    checkpointer = await get_checkpointer()
    checkpoint = await checkpointer.aget(config)

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
