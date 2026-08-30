from typing import TypedDict, Any


class RAGState(TypedDict, total=False):

    # Original user question
    question: str

    # Question after rewriting
    rewritten_question: str

    # Retrieved chunks
    documents: list[Any]

    # Combined context
    context: str

    # Generated answer
    answer: str

    # Retrieval quality
    documents_relevant: bool

    # Number of retrieval attempts
    retry_count: int

    # Observability helper
    observer: Any
