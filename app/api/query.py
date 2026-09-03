from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_rag_graph, get_hybrid_rag_graph
from app.models.schemas import QueryRequest, UserInDB
from app.observability import RAGObserver, RAGEvaluator
from app.services.auth.dependencies import get_current_user

router = APIRouter(prefix="/query", tags=["Query"])


def _invoke_graph(rag_graph, question: str, user_id: str) -> dict:
    """Shared invocation logic for both query endpoints."""
    observer = RAGObserver()
    evaluator = RAGEvaluator()

    result = rag_graph.invoke(
        {
            "question": question,
            "retry_count": 0,
            "user_id": user_id,
        },
        config={
            "run_name": "rag-query",
            "tags": ["rag", "corrective-rag"],
            "metadata": {"source": "fastapi", "user_id": user_id},
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

    return _invoke_graph(rag_graph, request.question, current_user.id)


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

    return _invoke_graph(rag_graph, request.question, current_user.id)
