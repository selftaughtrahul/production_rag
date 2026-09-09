from typing import TypedDict, Any, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages



class RAGState(TypedDict, total=False):
    question: str               # original question from the user
    rewritten_question: str     # query rewritten by LLM for better retrieval
    documents: list[Any]        # retrieved + reranked chunks
    context: str                # formatted context string passed to LLM
    answer: str                 # final answer from LLM
    documents_relevant: bool    # grading result: True → generate, False → rewrite
    retry_count: int            # number of retrieve-rewrite loops so far
    output_retry_count: int     # number of output-guardrail retries
    user_id: str                # authenticated user ID — used for metadata filtering
    chat_history: Annotated[list[BaseMessage], add_messages]  # HumanMessage / AIMessage objects
    long_term_memories: list[Any] # retrieved facts about the user
    skip_generate: bool         # retrieval-only invoke (streaming path)
    skip_memory_persist: bool   # skip save_memory when nested under the master agent

    error: str | None           # error message if any
    error_node: str | None      # node where error occurred
    error_type: str | None      # type of error (e.g., "llm", "retrieval")
    has_error: bool             # convenience flag
    error_retry_count: int      # number of error retries

    input_guardrail_passed: bool 
    input_guardrail_reason: str

    output_guardrail_passed: bool
    output_guardrail_reason: str

    guardrail_metadata: dict[str, Any]


