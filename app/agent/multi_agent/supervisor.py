import logging
from typing import Any, Dict, List, Literal
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from app.agent.multi_agent.route import next_agent
from app.agent.multi_agent.state import AgentOutput, SupervisorState
from app.services.llm.claude import ClaudeService

logger = logging.getLogger(__name__)


class SupervisorDecision(BaseModel):
    next: Literal["rag_agent", "sql_agent", "web_agent", "general_agent", "FINISH"] = Field(
        ...,
        description="Select the specialized sub-agent to run next, or 'FINISH' if you have gathered all necessary information.",
    )
    reasoning: str = Field(..., description="Brief explanation for the routing decision.")


SUPERVISOR_ROUTING_PROMPT = """\
You are an Enterprise Multi-Agent Supervisor orchestrating specialized specialist agents to solve user queries.

Specialist Agents:
1. 'rag_agent': Searches internal company documents, knowledge base, uploaded PDFs, policies, and domain documentation.
2. 'sql_agent': Inspects internal database tables, schemas, and executes read-only SQL queries (e.g., user counts, orders, system tables).
3. 'web_agent': Conducts live internet web search for breaking news, current public events, real-time facts, or external entities.
4. 'general_agent': Handles greetings, pleasantries, creative tasks, math, coding logic, or questions that can be answered from general LLM knowledge WITHOUT external search.
5. 'FINISH': Choose this when you have sufficient information from previous agent observations to answer the user's question, or if no further sub-agent calls are needed.

Decision Rules:
- If this is the start and the query is a simple greeting or general knowledge question, select 'general_agent' or 'FINISH'.
- If the query asks about internal documents or company policies, select 'rag_agent'.
- If the query asks about database tables, metrics, or row counts, select 'sql_agent'.
- If the query asks about current events, news, or external public facts, select 'web_agent'.
- If the user query is multi-part, dispatch to the first relevant specialist. Once that specialist returns with an observation, you can dispatch to the next specialist, or choose 'FINISH'.
- Do NOT call the same agent multiple times with the exact same query if it already returned a valid result.
"""

SUPERVISOR_SYNTHESIS_PROMPT = """\
You are an expert Enterprise AI Assistant.
Synthesize a comprehensive, clear, and accurate final answer for the user based on the conversation history, user profile, and the observations collected from the specialized agents.

Guidelines:
- If specialized agent observations are available (RAG documents, SQL queries, or Web search results), cite and integrate them directly.
- Use long-term memories about the user when relevant to personalize the response.
- If an agent encountered an error or found no results, be transparent and explain what was checked.
- Provide a direct, professional, and well-structured answer.
"""


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
                system_prompt=SUPERVISOR_ROUTING_PROMPT,
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
            system_prompt=SUPERVISOR_SYNTHESIS_PROMPT,
            max_tokens=1500,
        )

        return {
            "final_response": final_text,
            "messages": [AIMessage(content=final_text)],
        }