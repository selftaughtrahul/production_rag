from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_rag_graph
from app.models.schemas import QueryRequest, UserInDB
from app.observability import RAGObserver, RAGEvaluator
from app.services.auth.dependencies import get_current_user

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("/")
async def query_documents(
    request: QueryRequest,
    rag_graph=Depends(get_rag_graph),
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Ask a question against the authenticated user's documents.

    The user_id from the JWT is injected into the RAG graph state so that
    ChromaDB retrieval is filtered to only that user's chunks.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    observer = RAGObserver()
    evaluator = RAGEvaluator()

    result = rag_graph.invoke(
        {
            "question": request.question,
            "retry_count": 0,
            "observer": observer,
            "user_id": current_user.id,   # ← scoped retrieval
        },
        config={
            "run_name": "rag-query",
            "tags": ["rag", "corrective-rag"],
            "metadata": {"source": "fastapi", "user_id": current_user.id},
        },
    )

    metrics = observer.finish()
    evaluation = evaluator.evaluate(
        question=request.question,
        answer=result.get("answer", ""),
        context=result.get("context", ""),
    ).as_dict()

    return {
        "success": True,
        "question": request.question,
        "answer": result.get("answer", ""),
        "metrics": metrics,
        "evaluation": evaluation,
    }
