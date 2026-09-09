import logging
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field
from app.services.retriever.hybrid import HybridRetriever
from app.tools.base import BaseAgentTool, ToolResult

logger = logging.getLogger(__name__)


class DocumentSearchInput(BaseModel):
    query: str = Field(
        ..., description="Search query string to search internal vector/keyword document database."
    )
    top_k: int = Field(
        default=5, ge=1, le=20, description="Number of document chunks to return."
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

    def _run(self, query: str, top_k: int = 5) -> str:
        """Synchronous execution wrapper."""
        try:
            results = self.retriever.retrieve(query=query, top_k=top_k)
            formatted_docs = []
            for i, doc in enumerate(results, 1):
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "N/A")
                score = getattr(doc, "score", 0.0)
                formatted_docs.append(
                    f"[{i}] Document: {source} (Page {page}) [Relevance Score: {score:.2f}]\nContent: {doc.page_content}"
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

    async def _arun(self, query: str, top_k: int = 5) -> str:
        """Async execution bridge."""
        return self._run(query=query, top_k=top_k)
