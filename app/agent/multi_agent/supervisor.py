import logging
import re
from typing import Any, Dict, List, Literal
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from app.agent.multi_agent.route import next_agent
from app.agent.multi_agent.state import AgentOutput, SupervisorState
from app.prompts.specialist_prompts import (
    SUPERVISOR_ROUTING_V2,
    SUPERVISOR_SYNTHESIS_V2,
)
from app.services.llm.claude import ClaudeService

logger = logging.getLogger(__name__)

_ORDER_CONTROL_RE = re.compile(
    r"^(?:confirm|cancel) order act_[a-f0-9]{8}$",
    re.IGNORECASE,
)
_UNRELATED_PENDING_RE = re.compile(
    r"^(?:what|why|who|when|where|how|tell|explain|write|code|hello|hi|thanks|"
    r"search|calculate|compute|"
    r"show my orders?|list my orders?|get order|read (?:the )?document)",
    re.IGNORECASE,
)


def _pending_turn_targets_sql(state: SupervisorState) -> bool:
    pending = state.get("pending_order_action")
    if not pending:
        return False
    query = state.get("query", "").strip()
    if _ORDER_CONTROL_RE.fullmatch(query):
        return True
    if pending.get("phase") == "awaiting_confirmation":
        return query.lower() in {"yes", "no", "approve", "confirm", "cancel"}
    return not bool(_UNRELATED_PENDING_RE.match(query))


class SupervisorDecision(BaseModel):
    next: Literal["rag_agent", "sql_agent", "web_agent", "general_agent", "FINISH"] = Field(
        ...,
        description="Select the specialized sub-agent to run next, or 'FINISH' if you have gathered all necessary information.",
    )
    reasoning: str = Field(..., description="Brief explanation for the routing decision.")


class SupervisorNode:
    """Master router and synthesizer directing specialized sub-agents."""

    def __init__(self, llm: ClaudeService):
        self.llm = llm

    async def __call__(self, state: SupervisorState) -> Dict[str, Any]:
        """Evaluates state and decides the next node or FINISH."""
        iterations = state.get("iterations", 0) + 1
        max_iterations = state.get("max_iterations", 5)

        if iterations > max_iterations:
            logger.warning(f"Supervisor reached max iterations ({max_iterations}). Routing to FINISH.")
            return {
                "next_node": "FINISH",
                "iterations": iterations,
            }

        query = state.get("query", "")
        agent_outputs: List[AgentOutput] = state.get("agent_outputs") or []
        if _pending_turn_targets_sql(state) and not agent_outputs:
            logger.info("Pending order action routes directly to sql_agent")
            return {"next_node": "sql_agent", "iterations": iterations}
        already_ran = [
            out.agent_name for out in agent_outputs if out.agent_name != "__reset__"
        ]
        cheap = next_agent(query, already_ran)
        if cheap:
            logger.info("Supervisor cheap route -> %s", cheap)
            return {"next_node": cheap, "iterations": iterations}

        memories = state.get("long_term_memories") or []
        memories_str = "\n".join(f"- {m}" for m in memories) or "(none)"
        outputs_str = ""
        if agent_outputs:
            outputs_str = "\n\n".join(
                f"[{out.agent_name} Observation]:\n{out.result}" for out in agent_outputs
            )
        else:
            outputs_str = "(No agent observations yet)"

        prompt = f"""\
User Query: {query}

User Long-Term Profile:
{memories_str}

Agent Observations so far:
{outputs_str}

Current Iteration: {iterations}/{max_iterations}

Determine the next step:"""

        try:
            decision: SupervisorDecision = await self.llm.agenerate_structured(
                prompt=prompt,
                system_prompt=SUPERVISOR_ROUTING_V2,
                response_model=SupervisorDecision,
            )
            logger.info(f"Supervisor routed to [{decision.next}]. Reason: {decision.reasoning}")
            next_node = decision.next
        except Exception as e:
            logger.error(f"Supervisor decision failed: {e}. Defaulting to 'general_agent'.")
            next_node = "general_agent" if not agent_outputs else "FINISH"

        return {
            "next_node": next_node,
            "iterations": iterations,
        }

    async def synthesize(self, state: SupervisorState) -> Dict[str, Any]:
        """Use the single specialist answer as-is. Call the LLM only when several agents ran."""
        query = state.get("query", "")
        agent_outputs: List[AgentOutput] = state.get("agent_outputs") or []
        if len(agent_outputs) == 1 and (agent_outputs[0].result or "").strip():
            final_text = agent_outputs[0].result.strip()
            return {
                "final_response": final_text,
                "messages": [AIMessage(content=final_text)],
            }

        memories = state.get("long_term_memories") or []
        memories_str = "\n".join(f"- {m}" for m in memories) or "(none)"

        observations_str = ""
        if agent_outputs:
            observations_str = "\n\n".join(
                f"=== {out.agent_name} Output ===\n{out.result}" for out in agent_outputs
            )
        else:
            observations_str = "(Direct response, no external tools called)"

        prompt = f"""\
Original User Question: {query}

User Long-term Memories:
{memories_str}

Collected Specialist Findings:
{observations_str}

Synthesize the final answer for the user:"""

        final_text = await self.llm.agenerate(
            prompt=prompt,
            system_prompt=SUPERVISOR_SYNTHESIS_V2,
            max_tokens=1500,
        )

        return {
            "final_response": final_text,
            "messages": [AIMessage(content=final_text)],
        }