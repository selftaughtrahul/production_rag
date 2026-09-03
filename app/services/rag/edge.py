from .state import RAGState


def decide_after_grading(state: RAGState):

    relevant = state.get("documents_relevant",False,)
    retry_count = state.get( "retry_count", 0,)

    if relevant:
        return "generate"

    if retry_count >= 2:
        return "generate"

    return "rewrite"
