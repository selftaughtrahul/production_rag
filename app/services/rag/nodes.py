"""
RAG graph node implementations.

Each method accepts the current RAGState and returns a dict of
updated state fields. LangGraph merges the returned dict back
into the shared state automatically.
"""

from __future__ import annotations

import logging

from app.core.config import Settings
from .state import RAGState

logger = logging.getLogger(__name__)


class RAGNodes:
    """
    Collection of all node functions used inside the RAG StateGraph.

    Nodes (in execution order):
        retrieve        → fetch candidate chunks from the vector store
        rerank          → cross-encoder re-scoring of retrieved chunks
        grade_documents → decide if chunks are relevant enough
        rewrite_query   → LLM rewrites the question for a better search
        build_context   → format chunks into a single context string
        generate        → LLM produces the final answer
    """

    def __init__(self, retriever, reranker, context_builder, llm) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.context_builder = context_builder
        self.llm = llm

    # ──────────────────────────────────────────────────────────────────
    # Node 1 — Retrieve
    # ──────────────────────────────────────────────────────────────────

    def retrieve(self, state: RAGState) -> dict:
        """
        Fetch the top-N candidate chunks from the vector store.

        Scoped to the authenticated user via metadata_filter so that
        users never see each other's documents.
        """
        observer = state.get("observer")
        if observer:
            observer.on_retrieval_start()

        question = state["question"]
        query = state.get("rewritten_question") or question
        retry_count = state.get("retry_count", 0)

        user_id = state.get("user_id")
        metadata_filter = {"user_id": user_id} if user_id else None

        documents = self.retriever.retrieve(
            query=query,
            top_k=20,
            metadata_filter=metadata_filter,
        )

        logger.info("Retrieved %d chunks (user=%s, attempt=%d)", len(documents), user_id, retry_count + 1)

        if observer:
            observer.on_retrieval_end(document_count=len(documents))

        return {
            "documents": documents,
            "retry_count": retry_count + 1,
        }

    # ──────────────────────────────────────────────────────────────────
    # Node 2 — Rerank
    # ──────────────────────────────────────────────────────────────────

    def rerank(self, state: RAGState) -> dict:
        """
        Re-score retrieved chunks using a cross-encoder model.

        Returns the top-K most relevant chunks for grading.
        """
        question = state["question"]
        query = state.get("rewritten_question") or question
        documents = state.get("documents", [])

        if not documents:
            logger.warning("Reranker received 0 documents — skipping.")
            return {"documents": []}

        settings = Settings.from_environment()

        reranked = self.reranker.rerank(
            query=query,
            documents=documents,
            top_k=settings.rerank_top_k,
        )

        logger.info("Reranked: %d → %d documents", len(documents), len(reranked))
        return {"documents": reranked}

    # ──────────────────────────────────────────────────────────────────
    # Node 3 — Grade Documents
    # ──────────────────────────────────────────────────────────────────

    def grade_documents(self, state: RAGState) -> dict:
        """
        Decide whether the retrieved chunks are relevant enough to answer.

        Uses the rerank_score set by the cross-encoder.
        Any document with a score >= -1.0 is considered relevant
        (cross-encoder scores are negative; higher is better).
        """
        observer = state.get("observer")
        documents = state.get("documents", [])

        if not documents:
            logger.warning("No documents to grade — marking as not relevant.")
            if observer:
                observer.on_documents_graded(0)
            return {"documents_relevant": False}

        relevant = [
            doc for doc in documents
            if doc.metadata.get("rerank_score", 0.0) >= -1.0
        ]

        is_relevant = len(relevant) > 0
        final_docs = relevant if is_relevant else documents[:3]

        logger.info(
            "Grading: %d docs → %d relevant (relevant=%s)",
            len(documents), len(final_docs), is_relevant,
        )

        if observer:
            observer.on_documents_graded(len(final_docs))

        return {
            "documents": final_docs,
            "documents_relevant": is_relevant,
        }

    # ──────────────────────────────────────────────────────────────────
    # Node 4 — Rewrite Query
    # ──────────────────────────────────────────────────────────────────

    def rewrite_query(self, state: RAGState) -> dict:
        """
        Ask the LLM to rewrite the question for better semantic retrieval.

        Called when grade_documents decides the current results are not relevant.
        """
        observer = state.get("observer")
        question = state["question"]
        chat_history = state.get("chat_history", [])

        rewritten = self.llm.rewrite_query(question, chat_history=chat_history)

        logger.info("Query rewritten:\n  Before: %s\n  After:  %s", question, rewritten)

        if observer:
            observer.on_query_rewritten()
            observer.on_retry()

        return {"rewritten_question": rewritten}

    # ──────────────────────────────────────────────────────────────────
    # Node 5 — Build Context
    # ──────────────────────────────────────────────────────────────────

    def build_context(self, state: RAGState) -> dict:
        """
        Format the final set of documents into a single context string
        that the LLM can read.
        """
        documents = state.get("documents", [])

        if not documents:
            return {"context": ""}

        context = self.context_builder.build(documents)
        return {"context": context}

    # ──────────────────────────────────────────────────────────────────
    # Node 6 — Generate
    # ──────────────────────────────────────────────────────────────────

    def generate(self, state: RAGState) -> dict:
        """
        Generate the final answer using the LLM and the built context.
        """
        observer = state.get("observer")
        if observer:
            observer.on_generation_start()

        question = state["question"]
        context = state.get("context", "")
        chat_history = state.get("chat_history", [])

        if not context:
            logger.warning("No context available — returning fallback answer.")
            if observer:
                observer.on_generation_end()
            answer = "I could not find relevant information in the provided documents."
        else:
            answer = self.llm.generate(question=question, context=context, chat_history=chat_history)

        if observer:
            observer.on_generation_end()

        return {
            "answer": answer,
            "chat_history": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        }
