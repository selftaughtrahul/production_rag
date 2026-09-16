import logging
from langgraph.graph import StateGraph, START, END

from app.agent.multi_agent.state import SupervisorState
from app.agent.multi_agent.supervisor import SupervisorNode
from app.agent.multi_agent.nodes import (
    InputGuardrailNode,
    LoadMemoryNode,
    OutputGuardrailNode,
    SaveMemoryNode,
    ErrorHandlerNode,
)
from app.agent.multi_agent.specialists import (
    GeneralLLMNode,
    RAGSubGraphNode,
    SQLSubGraphNode,
    WebAgentNode,
)
from app.guardrails.input_service import InputGuardrailService
from app.guardrails.output_service import OutputGuardrailService
from app.services.llm.claude import ClaudeService
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def build_master_agent_graph(
    llm: ClaudeService,
    tool_registry: ToolRegistry,
    compiled_rag_graph=None,
    input_guardrails: InputGuardrailService | None = None,
    output_guardrails: OutputGuardrailService | None = None,
    checkpointer=None,
):
    """
    Builds the Master Multi-Agent Orchestration Graph with Supervisor-level
    guardrails (front-door input & exit-door output) and unified memory
    (LangGraph checkpointer + SQLite long-term user facts).
    """
    master_graph = StateGraph(SupervisorState)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Instantiate Nodes
    # ─────────────────────────────────────────────────────────────────────────
    input_guardrail_node = InputGuardrailNode(guardrail_service=input_guardrails)
    load_memory_node = LoadMemoryNode()
    supervisor = SupervisorNode(llm=llm)

    rag_node = RAGSubGraphNode(
        compiled_rag_graph=compiled_rag_graph,
        tool_registry=tool_registry,
    )
    sql_node = SQLSubGraphNode(llm=llm, tool_registry=tool_registry)
    web_node = WebAgentNode(llm=llm, tool_registry=tool_registry)
    general_node = GeneralLLMNode(llm=llm, tool_registry=tool_registry)

    output_guardrail_node = OutputGuardrailNode(guardrail_service=output_guardrails)
    save_memory_node = SaveMemoryNode(llm=llm)
    error_handler_node = ErrorHandlerNode()

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Add Nodes to Graph
    # ─────────────────────────────────────────────────────────────────────────
    master_graph.add_node("validate_input", input_guardrail_node)
    master_graph.add_node("load_memory", load_memory_node)
    master_graph.add_node("supervisor", supervisor)

    master_graph.add_node("rag_agent", rag_node)
    master_graph.add_node("sql_agent", sql_node)
    master_graph.add_node("web_agent", web_node)
    master_graph.add_node("general_agent", general_node)

    master_graph.add_node("synthesize", supervisor.synthesize)
    master_graph.add_node("validate_output", output_guardrail_node)
    master_graph.add_node("save_memory", save_memory_node)
    master_graph.add_node("error_handler", error_handler_node)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Define Graph Edges & Routing
    # ─────────────────────────────────────────────────────────────────────────

    # Entry point: START -> validate_input
    master_graph.add_edge(START, "validate_input")

    # Conditional after Input Guardrails: continue to memory or block to error_handler
    def decide_after_input(state: SupervisorState) -> str:
        if not state.get("input_guardrail_passed", True):
            return "blocked"
        return "continue"

    master_graph.add_conditional_edges(
        "validate_input",
        decide_after_input,
        {
            "continue": "load_memory",
            "blocked": "error_handler",
        },
    )

    # load_memory -> supervisor
    master_graph.add_edge("load_memory", "supervisor")

    # Conditional routing from Supervisor to specialists or synthesis
    def decide_from_supervisor(state: SupervisorState) -> str:
        next_step = state.get("next_node", "FINISH")
        if next_step in ["rag_agent", "sql_agent", "web_agent", "general_agent", "FINISH"]:
            return next_step
        return "FINISH"

    master_graph.add_conditional_edges(
        "supervisor",
        decide_from_supervisor,
        {
            "rag_agent": "rag_agent",
            "sql_agent": "sql_agent",
            "web_agent": "web_agent",
            "general_agent": "general_agent",
            "FINISH": "synthesize",
        },
    )

    # Specialist agents all cycle observations back to supervisor
    master_graph.add_edge("rag_agent", "supervisor")
    master_graph.add_edge("sql_agent", "supervisor")
    master_graph.add_edge("web_agent", "supervisor")
    master_graph.add_edge("general_agent", "supervisor")

    # Synthesis -> Output Guardrail check
    master_graph.add_edge("synthesize", "validate_output")

    # Conditional after Output Guardrails
    def decide_after_output(state: SupervisorState) -> str:
        if not state.get("output_guardrail_passed", True):
            return "blocked"
        return "continue"

    master_graph.add_conditional_edges(
        "validate_output",
        decide_after_output,
        {
            "continue": "save_memory",
            "blocked": "error_handler",
        },
    )

    # save_memory and error_handler terminate the turn
    master_graph.add_edge("save_memory", END)
    master_graph.add_edge("error_handler", END)

    logger.info("Compiled Master Multi-Agent Graph with Supervisor guardrails and memory.")
    return master_graph.compile(checkpointer=checkpointer)