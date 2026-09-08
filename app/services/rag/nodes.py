from anthropic.types import browser_get_page_text_config_param
from anthropic.types import browser_get_page_text_config_param
from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, AIMessage

from app.core.config import Settings
from app.memory.extractor import MemoryExtractor
from app.memory.service import MemoryService
from database.sqlite import get_connection
from .state import RAGState
from app.core.exceptions import handle_node_error


logger = logging.getLogger(__name__)


class RAGNodes:
    """
    Collection of all node functions used inside the RAG StateGraph.

    Nodes (in execution order):
        load_memory     → load persisted facts about the user
        retrieve        → fetch candidate chunks from the vector store
        rerank          → cross-encoder re-scoring of retrieved chunks
        grade_documents → decide if chunks are relevant enough
        rewrite_query   → LLM rewrites the question for a better search
        build_context   → format chunks into a single context string
        generate        → LLM produces the final answer
        save_memory     → extract and persist new long-term memories
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
        """ Retrieve documents from the vector store """
        try:
            observer = state.get("observer")

            if observer:
                observer.on_retrieval_start()

            question = state["question"]
            query = state.get("rewritten_question") or question

            user_id = state.get("user_id")

            metadata_filter = (
                {"user_id": user_id}
                if user_id
                else None
            )

            documents = self.retriever.retrieve(
                query=query,
                top_k=20,
                metadata_filter=metadata_filter,
            )

            logger.info(
                "Retrieved %d chunks (user=%s)",
                len(documents),
                user_id,
            )

            if observer:
                observer.on_retrieval_end(
                    document_count=len(documents)
                )

            return {
                "documents": documents,
                "has_error": False,
                "error": None,
                "error_node": None,
                "error_type": None,
            }

        except Exception as exc:
            logger.exception("Retrieval failed")
            return handle_node_error(state,"retrieve",exc)

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
        try:
            observer = state.get("observer")
            question = state["question"]

            retry_count = state.get("retry_count", 0)

            # Convert LangChain messages → plain dicts for Claude API
            history_dicts = [
                {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
                for m in state.get("chat_history", [])
            ]

            rewritten = self.llm.rewrite_query(question, chat_history=history_dicts)

            logger.info("Query rewritten:\n  Before: %s\n  After:  %s", question, rewritten)

            if observer:
                observer.on_query_rewritten()
                observer.on_retry()

            return {
                "rewritten_question": rewritten,
                "retry_count": retry_count + 1,
                "has_error": False,
                "error": None,
                "error_node": None,
                "error_type": None,
            }
        except Exception as exc:

            return handle_node_error(
                state,
                "rewrite",
                exc,
            )

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
    # Node 6 — Load long-term memory
    # ──────────────────────────────────────────────────────────────────

    def load_memory(self, state: RAGState) -> dict:
        """Load persisted facts about the authenticated user."""
        user_id = state.get("user_id")
        if not user_id:
            return {"long_term_memories": []}

        conn = get_connection()
        try:
            memories = MemoryService(conn).get_user_memories(user_id=user_id, limit=10)
        finally:
            conn.close()

        logger.info("Loaded %d long-term memories (user=%s)", len(memories), user_id)
        return {"long_term_memories": [item.memory for item in memories]}

    # ──────────────────────────────────────────────────────────────────
    # Node 7 — Generate
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

        # Convert LangChain messages → plain dicts for Claude API
        history_dicts = [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
            for m in state.get("chat_history", [])
        ]

        if not context:
            logger.info("No context available — LLM will answer from chat history if possible.")

        memories = state.get("long_term_memories") or []
        answer = self.llm.generate(
            question=question,
            context=context,
            chat_history=history_dicts,
            long_term_memories=memories,
        )

        if observer:
            observer.on_generation_end()

        return {
            "answer": answer,
            # Return proper LangChain message objects — required by the add_messages reducer
            "chat_history": [HumanMessage(content=question), AIMessage(content=answer)],
        }

    # ──────────────────────────────────────────────────────────────────
    # Node 8 — Save long-term memory
    # ──────────────────────────────────────────────────────────────────

    def save_memory(self, state: RAGState) -> dict:
        """Extract durable facts from this turn and persist them."""
        user_id = state.get("user_id")
        question = state.get("question", "")
        answer = state.get("answer", "")

        if not user_id or not question or not answer:
            return {}

        conversation = f"User: {question}\nAssistant: {answer}"
        conn = get_connection()
        try:
            memory_service = MemoryService(conn)
            existing = memory_service.get_user_memories(user_id=user_id, limit=20)
            decision = MemoryExtractor(self.llm).decide(
                user_id=user_id,
                conversation=conversation,
                existing_memories=existing,
            )

            if decision.action == "ADD" and decision.memory:
                memory_service.create_memory(
                    user_id=user_id,
                    memory=decision.memory,
                    memory_type=decision.memory_type or "general",
                    importance=decision.importance,
                )
                logger.info("Stored long-term memory for user=%s: %s", user_id, decision.memory)
            elif decision.action == "UPDATE" and decision.memory and decision.memory_id:
                updated = memory_service.update_memory(
                    memory_id=decision.memory_id,
                    memory=decision.memory,
                    memory_type=decision.memory_type or "general",
                    importance=decision.importance,
                    user_id=user_id,
                )
                if updated:
                    logger.info("Updated long-term memory %s for user=%s", decision.memory_id, user_id)
                else:
                    logger.warning(
                        "Memory UPDATE skipped; id %s not found for user=%s",
                        decision.memory_id,
                        user_id,
                    )
            else:
                logger.info("No long-term memory change (action=%s)", decision.action)
        except Exception:
            logger.exception("Failed to persist long-term memory for user=%s", user_id)
        finally:
            conn.close()

        return {}

    def error_handler(self, state: RAGState):
        """Handle errors that occur in any RAG pipeline node."""

        error_node = state.get("error_node")
        error = state.get("error")

        logger.error(f"RAG Error | node={error_node} | error={error}")

        return {"has_error": True}
