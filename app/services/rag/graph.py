from langgraph.graph import StateGraph, END, START
from .state import RAGState
from .nodes import RAGNodes
from .edge import (
    decide_after_input_guardrail,
    decide_after_grading,
    decide_after_error,
    decide_after_output_guardrail,
)


def build_rag_graph(retriever, reranker, context_builder, llm,input_guardrails=None, output_guardrails=None, checkpointer=None):

    nodes = RAGNodes(
        retriever=retriever,
        reranker=reranker,
        context_builder=context_builder,
        llm=llm,
        input_guardrail=input_guardrails,
        output_guardrail=output_guardrails



    )

    graph = StateGraph(RAGState)


    # Nodes
    graph.add_node("validate_input",nodes.validate_input)
    graph.add_node("load_memory", nodes.load_memory)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("reranker", nodes.rerank)
    graph.add_node("grade_documents", nodes.grade_documents)
    graph.add_node("rewrite", nodes.rewrite_query)
    graph.add_node("build_context", nodes.build_context)
    graph.add_node("generate", nodes.generate)
    graph.add_node("validate_output",nodes.validate_output)
    graph.add_node("save_memory", nodes.save_memory)
    graph.add_node("error_handler", nodes.error_handler)

    # Start
    graph.add_edge(
        START,
        "validate_input",
    )
    graph.add_conditional_edges(
        "validate_input",
        decide_after_input_guardrail,
        {
            "continue": "load_memory",
            "blocked": "error_handler",
        }
        
        
    )

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
   
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("build_context", "generate")
    graph.add_edge("generate", "validate_output")
    graph.add_conditional_edges(
        "validate_output",
        decide_after_output_guardrail,
        {
            "continue": "save_memory",
            "retry": "generate",
            "error": "error_handler",
        },
    )

    graph.add_edge("save_memory", END)
    graph.add_edge("error_handler", END)

    return graph.compile(checkpointer=checkpointer)