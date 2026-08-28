from __future__ import annotations
from app.llm.claude import ClaudeService
from app.retriever.context import ContextBuilder
from app.retriever.service import RetrieverService


class QueryService:
    """
    Complete RAG query service.

    Flow:
        Query
          ↓
        Retriever
          ↓
        Context Builder
          ↓
        Claude
          ↓
        Answer
    """

    def __init__(
        self,
        retriever: RetrieverService,
        context_builder: ContextBuilder,
        llm: ClaudeService,
    ) -> None:

        self.retriever = retriever
        self.context_builder = context_builder
        self.llm = llm

    def query(
        self,
        question: str,
        top_k: int = 5,
    ) -> dict:

        # -----------------------------------------
        # 1. Retrieve relevant chunks
        # -----------------------------------------

        chunks = self.retriever.retrieve(
            query=question,
            top_k=top_k,
        )

        # -----------------------------------------
        # 2. Build context
        # -----------------------------------------

        context = self.context_builder.build(chunks)

        # -----------------------------------------
        # 3. Generate answer using Claude
        # -----------------------------------------

        answer = self.llm.generate(
            question=question,
            context=context,
        )

        return {
            "answer": answer,
            "sources": [
                {
                    "score": chunk.score,
                    "metadata": chunk.metadata,
                }
                for chunk in chunks
            ],
        }
