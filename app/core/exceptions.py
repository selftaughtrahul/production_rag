from app.services.rag.state import RAGState


class LLMUnavailableError(RuntimeError):
    """The LLM provider refused the call (quota, auth, or connectivity).

    Carries a message that is safe to show the user, so callers can report the
    real reason instead of a generic failure.
    """


def handle_node_error(state: RAGState, node_name: str, exc: Exception) -> dict:
    """
    Handle errors that occur in any RAG pipeline node.

    Args:
        state: current RAGState
        node_name: name of the node where error occurred
        exc: exception object

    Returns:
        updated RAGState with error information
    """

    retry_count = state.get("retry_count", 0)

    return {
        "has_error": True,
        "error": str(exc),
        "error_node": node_name,
        "error_type": type(exc).__name__,
        "retry_count": retry_count + 1,
    }