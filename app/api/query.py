from __future__ import annotations

import json
import sqlite3
import uuid

from fastapi import APIRouter, Depends, HTTPException
from langgraph.checkpoint.sqlite import SqliteSaver

from app.api.dependencies import get_rag_graph, get_hybrid_rag_graph
from app.models.schemas import QueryRequest, UserInDB
from app.observability import RAGObserver, RAGEvaluator
from app.services.auth.dependencies import get_current_user
from database.sqlite import get_connection

router = APIRouter(prefix="/query", tags=["Query"])



def generate_new_session_id() -> str:
    return f"session_{uuid.uuid4()}"


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
            "run_name": "rag-query",
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

    with SqliteSaver.from_conn_string("rag_database.db") as checkpointer:
        checkpoint = checkpointer.get(config)

        if not checkpoint:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # chat_history holds LangChain HumanMessage / AIMessage objects
        messages = checkpoint.get("channel_values", {}).get("chat_history", [])

        return {
            "success": True,
            "session_id": session_id,
            "messages": [
                {"role": m.type, "content": m.content}
                for m in messages
            ],
        }
