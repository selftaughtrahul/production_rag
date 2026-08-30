from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.dependencies import get_rag_graph
from app.observability import RAGObserver, RAGEvaluator

router = APIRouter(prefix="/query", tags=["Query"])


class QueryRequest(BaseModel):
    question: str


@router.post("/")
async def query_documents(
    request: QueryRequest,
    rag_graph=Depends(get_rag_graph),
):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    observer = RAGObserver()
    evaluator = RAGEvaluator()

    result = rag_graph.invoke(
        {
            "question": request.question,
            "retry_count": 0,
            "observer": observer,
        },
        config={
            "run_name": "rag-query",
            "tags": [
                "rag",
                "corrective-rag",
            ],
            "metadata": {
                "source": "fastapi",
            },
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
