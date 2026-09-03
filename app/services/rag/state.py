import operator
from typing import TypedDict, Any, Annotated
from langgraph.graph.message import add_messages



class RAGState(TypedDict, total=False):
    question: str               # original question from the user
    rewritten_question: str     # query rewritten by LLM for better retrieval
    documents: list[Any]        # retrieved + reranked chunks
    context: str                # formatted context string passed to LLM
    answer: str                 # final answer from LLM
    documents_relevant: bool    # grading result: True → generate, False → rewrite
    retry_count: int            # number of retrieve-rewrite loops so far
    observer: Any               # RAGObserver instance for metrics
    user_id: str                # authenticated user ID — used for metadata filtering
    # chat_history: Annotated[list[dict[str, str]], operator.add]  # appended at each step
    chat_history: Annotated[list, add_messages]

