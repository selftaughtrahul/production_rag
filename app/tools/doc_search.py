import asyncio
import logging
from typing import Type

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema
from sqlalchemy import select

from database.models import Document
from database.sqlite import SessionLocal
from app.services.retriever.hybrid import HybridRetriever
from app.tools.base import BaseAgentTool

logger = logging.getLogger(__name__)


class DocumentSearchInput(BaseModel):
    query: str = Field(
        ..., description="Search query string to search internal vector/keyword document database."
    )
    top_k: int = Field(
        default=5, ge=1, le=20, description="Number of document chunks to return."
    )
    user_id: SkipJsonSchema[str] = Field(
        ..., description="Authenticated user id; always provided by the orchestrator."
    )


class DocumentSearchTool(BaseAgentTool):
    name: str = "document_search"
    description: str = (
        "Searches internal company knowledge base, uploaded PDFs, MD, TXT, and DOCX files. "
        "Use this tool when answering questions about internal domain information, user documents, or uploaded files."
    )
    args_schema: Type[BaseModel] = DocumentSearchInput
    retriever: HybridRetriever

    class Config:
        arbitrary_types_allowed = True

    def _run(self, query: str, top_k: int = 5, user_id: str = "") -> str:
        """Synchronous execution wrapper."""
        if not user_id:
            return self._format_error(
                "Document search requires an authenticated user."
            ).to_str()
        try:
            results = self.retriever.retrieve(
                query=query,
                top_k=top_k,
                metadata_filter={"user_id": user_id},
            )
            formatted_docs = []
            for i, doc in enumerate(results, 1):
                source = doc.metadata.get("filename") or doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "N/A")
                score = getattr(doc, "score", 0.0)
                formatted_docs.append(
                    f"[{i}] Document: {source} (Page {page}) [Relevance Score: {score:.2f}]\nContent: {doc.text}"
                )

            if not formatted_docs:
                return self._format_success(
                    "No relevant internal documents found for the given query.",
                    metadata={"query": query, "count": 0},
                ).to_str()

            output = "\n\n---\n\n".join(formatted_docs)
            return self._format_success(
                output, metadata={"query": query, "count": len(results)}
            ).to_str()

        except Exception as e:
            logger.error(f"Error executing DocumentSearchTool: {str(e)}", exc_info=True)
            return self._format_error(f"Failed to execute document search: {str(e)}").to_str()

    async def _arun(self, query: str, top_k: int = 5, user_id: str = "") -> str:
        """Async execution bridge."""
        return await asyncio.to_thread(
            self._run,
            query=query,
            top_k=top_k,
            user_id=user_id,
        )


class ListUserDocumentsInput(BaseModel):
    user_id: SkipJsonSchema[str] = Field(min_length=1)
    limit: int = Field(default=50, ge=1, le=100)


class ListUserDocumentsTool(BaseAgentTool):
    name: str = "list_user_documents"
    description: str = "List uploaded documents owned by the authenticated user."
    args_schema: Type[BaseModel] = ListUserDocumentsInput

    def _run(self, user_id: str, limit: int = 50) -> str:
        with SessionLocal() as session:
            rows = session.execute(
                select(Document.id, Document.file_name, Document.created_at)
                .where(Document.user_id == user_id)
                .order_by(Document.created_at.desc())
                .limit(limit)
            ).all()
        documents = [
            {
                "id": document_id,
                "file_name": file_name,
                "created_at": created_at.isoformat() if created_at else None,
            }
            for document_id, file_name, created_at in rows
        ]
        return self._format_success(
            documents,
            metadata={"count": len(documents)},
        ).to_str()

    async def _arun(self, user_id: str, limit: int = 50) -> str:
        return await asyncio.to_thread(self._run, user_id, limit)
