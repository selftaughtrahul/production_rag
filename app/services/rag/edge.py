from .state import RAGState


def decide_after_grading(state: RAGState):
    """Decide the next node based on whether documents are relevant."""

    if state.get("has_error", False):
        return "error_handler"

    relevant = state.get("documents_relevant", False)
    retry_count = state.get("retry_count", 0)

    if relevant:
        return "build_context"

    if retry_count >= 2:
        return "build_context"

    return "rewrite"


def decide_after_error(state: RAGState):

    """Decide next node after an error occurs in the pipeline."""

    if not state.get("has_error", False):
        return "continue"

    retry_count = state.get("retry_count", 0)

    if retry_count < 2:
        return "retry"

    return "fallback"