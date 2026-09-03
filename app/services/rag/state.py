from typing import TypedDict, Any


class RAGState(TypedDict, total=False):
    question: str # orginal question
    rewritten_question: str
    documents: list[Any]
    context: str
    answer: str
    documents_relevant: bool
    retry_count: int
    observer: Any
