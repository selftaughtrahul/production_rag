"""
Collection of RAG pipeline node implementations.
"""

from langchain_core.messages import HumanMessage, AIMessage

from app.core.config import Settings
from app.core.exceptions import handle_node_error
from app.memory.persist import persist_from_turn
from app.memory.service import MemoryService
from database.sqlite import SessionLocal
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
            if not user_id:
                return {
                    "documents": [],
                    "has_error": False,
                    "error": None,
                    "error_node": None,
                    "error_type": None,
                }

            metadata_filter = {"user_id": user_id}
            settings = Settings.from_environment()
            # Cap pairs so the cross-encoder stays around 200ms, not a second pass of 20 chunks.
            retrieve_k = min(settings.dense_top_k, max(settings.rerank_top_k * 2, 8))

            documents = self.retriever.retrieve(
                query=query,
                top_k=retrieve_k,
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
        Decide whether retrieved chunks can be used to answer.

        The cross-encoder already ranked them. Do not apply a score cutoff
        that forces a 5s rewrite on every turn.
        """
        documents = state.get("documents", [])
        if not documents:
            return {"documents_relevant": False}

        return {
            "documents": documents,
            "documents_relevant": True,
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

        session = SessionLocal()
        try:
            memories = MemoryService(session).get_user_memories(user_id=user_id, limit=10)
        finally:
            session.close()

        return {"long_term_memories": [item.memory for item in memories]}

    # ──────────────────────────────────────────────────────────────────
    # Node 7 — Generate
    # ──────────────────────────────────────────────────────────────────

    def generate(self, state: RAGState) -> dict:
        """
        Generate the final answer using the LLM and the built context.
        """
 

        if state.get("skip_generate"):
            return {}

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
        if state.get("skip_memory_persist") or state.get("skip_generate"):
            return {}

        persist_from_turn(
            self.llm,
            state.get("user_id"),
            state.get("question", ""),
            state.get("answer", ""),
        )
        return {}

    def error_handler(self, state: RAGState):
        """Handle errors that occur in any RAG pipeline node."""
        return {
            "has_error": True,
            "answer": "I cannot complete your request right now.",
        }

    async def validate_input(self, state: RAGState):
        if self.input_guardrails is None:
            return {"input_guardrail_passed": True}

        result = await self.input_guardrails.validate(state["question"])
        return {
            "input_guardrail_passed": result.passed,
            "input_guardrail_reason": result.reason,
            "guardrail_metadata": result.metadata,
        }

    async def validate_output(self, state: RAGState):
        if state.get("skip_generate"):
            return {"output_guardrail_passed": True}

        if self.output_guardrails is None:
            return {"output_guardrail_passed": True}

        result = await self.output_guardrails.validate(
            query=state["question"],
            response=state.get("answer", ""),
            context=state.get("context"),
        )
        update = {
            "output_guardrail_passed": result.passed,
            "output_guardrail_reason": result.reason,
            "guardrail_metadata": {
                **state.get("guardrail_metadata", {}),
                **result.metadata,
            },
        }
        if not result.passed:
            update["output_retry_count"] = state.get("output_retry_count", 0) + 1
        return update
