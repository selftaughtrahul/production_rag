from app.services.rag.state import RAGState


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