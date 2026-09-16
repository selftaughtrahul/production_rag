from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class AgentOutput(BaseModel):
    """Container for individual sub-agent execution outputs."""

    agent_name: str
    result: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PendingOrderAction(BaseModel):
    """Checkpoint-safe order draft bound to one user and session."""

    action_id: str
    user_id: str
    session_id: str
    tool_name: Literal["create_order", "update_order"]
    payload: dict[str, Any] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    phase: Literal["collecting", "awaiting_confirmation"]
    expires_at: datetime
    idempotency_key: str
    payload_hash: str | None = None


def merge_agent_outputs(
    existing: List[AgentOutput] | None,
    incoming: List[AgentOutput] | None,
) -> List[AgentOutput]:
    """Append this turn's outputs. A __reset__ item drops the previous turn."""
    incoming = incoming or []
    if any(item.agent_name == "__reset__" for item in incoming):
        return [item for item in incoming if item.agent_name != "__reset__"]
    return list(existing or []) + list(incoming)


class SupervisorState(TypedDict, total=False):
    """Global state for the Supervisor Multi-Agent System."""

    query: str
    user_id: str
    thread_id: str

    # Conversational memory (LangGraph add_messages reducer)
    messages: Annotated[List[BaseMessage], add_messages]

    # Persistent long-term user facts (Postgres user_memories)
    long_term_memories: List[str]
    tool_approved: bool
    pending_order_action: dict[str, Any] | None

    # Next node to dispatch: 'rag_agent', 'sql_agent', 'web_agent', 'general_agent', 'FINISH'
    next_node: str

    # Trajectory of sub-agent outputs
    agent_outputs: Annotated[List[AgentOutput], merge_agent_outputs]

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