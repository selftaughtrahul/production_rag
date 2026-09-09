import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class AgentOutput(BaseModel):
    """Container for individual sub-agent execution outputs."""

    agent_name: str
    result: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SupervisorState(TypedDict, total=False):
    """Global state for the Supervisor Multi-Agent System."""

    query: str
    user_id: str
    thread_id: str

    # Conversational memory (LangGraph add_messages reducer)
    messages: Annotated[List[BaseMessage], add_messages]

    # Persistent long-term user facts loaded from SQLite
    long_term_memories: List[str]

    # Next node to dispatch: 'rag_agent', 'sql_agent', 'web_agent', 'general_agent', 'FINISH'
    next_node: str

    # Trajectory of sub-agent outputs
    agent_outputs: Annotated[List[AgentOutput], operator.add]

    # Synthesized final answer
    final_response: Optional[str]

    # Supervisor-level Input Guardrail flags
    input_guardrail_passed: bool
    input_guardrail_reason: Optional[str]

    # Supervisor-level Output Guardrail flags
    output_guardrail_passed: bool
    output_guardrail_reason: Optional[str]

    # Guardrail and observability metadata
    guardrail_metadata: Dict[str, Any]

    # Loop safeguards
    iterations: int
    max_iterations: int
    error: Optional[str]