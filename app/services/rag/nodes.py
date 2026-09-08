"""
Collection of RAG pipeline node implementations.
"""

from anthropic.types import browser_get_page_text_config_param
from langchain_core.messages import HumanMessage, AIMessage

from app.core.config import Settings
from app.core.exceptions import handle_node_error
from app.memory.extractor import MemoryExtractor
from app.memory.service import MemoryService
from database.sqlite import get_connection
from .state import RAGState





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
        error_handler   → handles errors in the pipeline


    """

    def __init__(self, retriever, reranker, context_builder, llm, input_guardrail=None, output_guardrail = None) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.context_builder = context_builder
        self.llm = llm
        self.input_guardrails = input_guardrail
        self.output_guardrails = output_guardrail

    # ──────────────────────────────────────────────────────────────────
    # Node 1 — Retrieve
    # ──────────────────────────────────────────────────────────────────

    def retrieve(self, state: RAGState) -> dict:
        """ Retrieve documents from the vector store """
        try:
           

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


            return {
                "documents": documents,
                "has_error": False,
                "error": None,
                "error_node": None,
                "error_type": None,
            }

        except Exception as exc:
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
            return {"documents": []}

        settings = Settings.from_environment()

        reranked = self.reranker.rerank(
            query=query,
            documents=documents,
            top_k=settings.rerank_top_k,
        )

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
      
        documents = state.get("documents", [])

        if not documents:
           
            return {"documents_relevant": False}

        relevant = [
            doc for doc in documents
            if doc.metadata.get("rerank_score", 0.0) >= -1.0
        ]

        is_relevant = len(relevant) > 0
        final_docs = relevant if is_relevant else documents[:3]



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
            question = state["question"]

            retry_count = state.get("retry_count", 0)

            # Convert LangChain messages → plain dicts for Claude API
            history_dicts = [
                {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
                for m in state.get("chat_history", [])
            ]

            rewritten = self.llm.rewrite_query(question, chat_history=history_dicts)


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

        return {"long_term_memories": [item.memory for item in memories]}

    # ──────────────────────────────────────────────────────────────────
    # Node 7 — Generate
    # ──────────────────────────────────────────────────────────────────

    def generate(self, state: RAGState) -> dict:
        """
        Generate the final answer using the LLM and the built context.
        """
 

        question = state["question"]
        context = state.get("context", "")

        # Convert LangChain messages → plain dicts for Claude API
        history_dicts = [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
            for m in state.get("chat_history", [])
        ]

       
        memories = state.get("long_term_memories") or []
        answer = self.llm.generate(
            question=question,
            context=context,
            chat_history=history_dicts,
            long_term_memories=memories,
        )

       

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
            elif decision.action == "UPDATE" and decision.memory and decision.memory_id:
                updated = memory_service.update_memory(
                    memory_id=decision.memory_id,
                    memory=decision.memory,
                    memory_type=decision.memory_type or "general",
                    importance=decision.importance,
                    user_id=user_id,
                )
               
            else:
                print("no change in long-term memory (action=%s)", decision.action)
        except Exception:
            print("Failed to persist long-term memory for user=%s", user_id)
        finally:
            conn.close()

        return {}

    def error_handler(self, state: RAGState):
        """Handle errors that occur in any RAG pipeline node."""

        error_node = state.get("error_node")
        error = state.get("error")


        return {"has_error": True}

    async def validate_input(self, state: RAGState):

        if self.input_guardrails is None:
            return state

        result = await self.input_guardrails.validate(
            state["query"]
        )

        return {
            **state,
            "input_guardrail_passed": result.passed,
            "input_guardrail_reason": result.reason,
            "guardrail_metadata": result.metadata,

        }
    
    async def validate_output(self, state: RAGState):

        if self.output_guardrails is None:
            return {
                **state,
                "output_guardrail_passed": True,
            }

        result = await self.output_guardrails.validate(
            query=state["query"],
            response=state["answer"],
            context=state.get("context"),
        )

        return {
            **state,
            "output_guardrail_passed": result.passed,
            "output_guardrail_reason": result.reason,
            "guardrail_metadata": {
                **state.get("guardrail_metadata", {}),
                **result.metadata,
            },
        }
