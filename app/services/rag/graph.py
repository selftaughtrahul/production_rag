from langgraph.graph import StateGraph, END, START
from .state import RAGState
from .nodes import RAGNodes
from .edge import decide_after_grading


def build_rag_graph(retriever, reranker, context_builder, llm):

    nodes = RAGNodes(retriever=retriever, reranker=reranker, context_builder=context_builder, llm=llm)
    graph = StateGraph(RAGState)

    # Nodes
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("grade_documents", nodes.grade_documents)
    graph.add_node("rewrite", nodes.rewrite_query)
    graph.add_node("build_context", nodes.build_context)
    graph.add_node("generate", nodes.generate)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {
            "generate": "build_context",
            "rewrite": "rewrite",
        },
    )

    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("build_context", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
