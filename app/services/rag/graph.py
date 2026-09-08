from langgraph.graph import StateGraph, END, START
from .state import RAGState
from .nodes import RAGNodes
from .edge import decide_after_grading, decide_after_error


def build_rag_graph(retriever, reranker, context_builder, llm, checkpointer=None):

    nodes = RAGNodes(
        retriever=retriever,
        reranker=reranker,
        context_builder=context_builder,
        llm=llm,
    )

    graph = StateGraph(RAGState)


    # Nodes
    graph.add_node("load_memory", nodes.load_memory)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("reranker", nodes.rerank)
    graph.add_node("grade_documents", nodes.grade_documents)
    graph.add_node("rewrite", nodes.rewrite_query)
    graph.add_node("build_context", nodes.build_context)
    graph.add_node("generate", nodes.generate)
    graph.add_node("save_memory", nodes.save_memory)
    graph.add_node("error_handler",nodes.error_handler)

    # Start
    graph.add_edge(START, "load_memory")

    # Memory → Retrieval
    graph.add_edge("load_memory", "retrieve")

    # Retrieval error handling
    graph.add_conditional_edges(
        "retrieve",
        decide_after_error,
        {
            "continue": "reranker",
            "retry": "retrieve",
            "fallback": "build_context",
        },
    )

    # Reranking
    graph.add_edge("reranker", "grade_documents")

    # Document grading
    graph.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {
            "build_context": "build_context",
            "rewrite": "rewrite",
            "error_handler": "error_handler",
        },
    )
   
    graph.add_edge("rewrite","retrieve")

    # ============================================================
    # CONTEXT → GENERATE
    # ============================================================

    graph.add_edge("build_context","generate")

    # ============================================================
    # GENERATE → SAVE MEMORY
    # ============================================================

    graph.add_edge("generate","save_memory")

    # ============================================================
    # END
    # ============================================================

    graph.add_edge("save_memory",END)

    # Error handler → END
    graph.add_edge("error_handler",END)

    return graph.compile(
        checkpointer=checkpointer,
    )