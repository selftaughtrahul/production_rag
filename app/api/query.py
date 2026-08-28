from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.dependencies import get_query_service
from app.query.service import QueryService

router = APIRouter(prefix="/query", tags=["Query"])


class QueryRequest(BaseModel):
    question: str


@router.post("/")
async def query_documents(
    request: QueryRequest,
    query_service: QueryService = Depends(get_query_service),
):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    result = query_service.query(question=request.question)
    return {"success": True, "question": request.question, **result}
