from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.dependencies import get_rag_graph
from app.query.service import QueryService

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

    result = rag_graph.invoke(
        {
            "question": request.question,
            "retry_count": 0,
        }
    )

    return {
        "success": True,
        "question": request.question,
        "answer": result["answer"],
    }
