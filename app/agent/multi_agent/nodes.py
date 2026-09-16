import logging
from typing import Any, Dict, List
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.multi_agent.state import AgentOutput, SupervisorState
from app.guardrails.input_service import InputGuardrailService
from app.guardrails.output_service import OutputGuardrailService
from app.memory.persist import persist_from_turn_background
from app.memory.service import MemoryService
from app.services.llm.claude import ClaudeService
from app.tools.registry import ToolRegistry
from database.sqlite import SessionLocal

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Input Guardrail Node (Front-Door Protection)
# ─────────────────────────────────────────────────────────────────────────────
class InputGuardrailNode:
    """Executes multi-layer input guardrails before Supervisor routing."""

    def __init__(self, guardrail_service: InputGuardrailService | None = None):
        self.guardrail_service = guardrail_service

    async def __call__(self, state: SupervisorState) -> Dict[str, Any]:
        # Same thread_id reuses the checkpoint. Clear last turn so Tesla
        # is not answered with the previous few-shot reply.
        update: Dict[str, Any] = {
            "agent_outputs": [AgentOutput(agent_name="__reset__", result="")],
            "iterations": 0,
            "final_response": None,
            "next_node": "",
            "input_guardrail_passed": True,
            "input_guardrail_reason": None,
        }

        query = state.get("query", "")
        if self.guardrail_service is None:
            return update

        logger.info("[InputGuardrail] Validating query before supervisor routing...")
        result = await self.guardrail_service.validate(query)

        if not result.passed:
            logger.warning(f"[InputGuardrail] Query BLOCKED: {result.reason}")

        update.update(
            {
                "input_guardrail_passed": result.passed,
                "input_guardrail_reason": result.reason,
                "guardrail_metadata": {
                    **state.get("guardrail_metadata", {}),
                    **result.metadata,
                    "input_action": result.action,
                },
            }
        )
        return update


# ─────────────────────────────────────────────────────────────────────────────
# 2. Load Memory Node (Supervisor-Level Long-Term Memory)
# ─────────────────────────────────────────────────────────────────────────────
class LoadMemoryNode:
    """Loads persistent user facts from SQLite user_memories into Supervisor state."""

    async def __call__(self, state: SupervisorState) -> Dict[str, Any]:
        user_id = state.get("user_id")
        if not user_id:
            return {"long_term_memories": []}

        session = SessionLocal()
        try:
            memory_service = MemoryService(session)
            memories = memory_service.get_user_memories(user_id=user_id, limit=10)
            facts = [m.memory for m in memories]
            logger.info(f"[LoadMemory] Loaded {len(facts)} long-term memories for user '{user_id}'")
            return {"long_term_memories": facts}
        except Exception as e:
            logger.warning(f"[LoadMemory] Failed to load memories for user '{user_id}': {e}")
            return {"long_term_memories": []}
        finally:
            session.close()


# ─────────────────────────────────────────────────────────────────────────────
# 3. RAG Sub-Graph Node (Specialized Document Retrieval)
# ─────────────────────────────────────────────────────────────────────────────
class RAGSubGraphNode:
    """Adapter node that invokes the compiled RAG StateGraph or internal DocumentSearchTool."""

    def __init__(self, compiled_rag_graph=None, tool_registry: ToolRegistry = None):
        self.rag_graph = compiled_rag_graph
        self.tool_registry = tool_registry

    async def __call__(self, state: SupervisorState) -> Dict[str, Any]:
        logger.info("[RAG Node] Executing internal knowledge retrieval...")
        query = state.get("query", "")
        user_id = state.get("user_id")
        thread_id = state.get("thread_id") or ""

        answer = ""
        context = ""

        if self.rag_graph:
            try:
                rag_input = {
                    "question": query,
                    "user_id": user_id,
                    "skip_memory_persist": True,
                    "skip_generate": True,
                    "rewritten_question": None,
                    "documents": [],
                    "context": "",
                    "answer": "",
                    "documents_relevant": False,
                    "has_error": False,
                    "retry_count": 0,
                }
                rag_config = {
                    "configurable": {
                        "thread_id": f"{thread_id}:rag" if thread_id else "rag",
                    }
                }
                rag_result = await self.rag_graph.ainvoke(rag_input, config=rag_config)
                context = rag_result.get("context", "")
                answer = rag_result.get("answer") or rag_result.get("generation", "")
                if context and not answer:
                    answer = "Retrieved internal documents."
            except Exception as e:
                logger.error(f"[RAG Node] RAG StateGraph invocation failed: {e}", exc_info=True)

        if not answer and self.tool_registry and user_id:
            doc_tool = self.tool_registry.get_tool("document_search")
            if doc_tool:
                doc_output = await doc_tool.ainvoke(
                    {"query": query, "top_k": 4, "user_id": user_id}
                )
                answer = f"Retrieved documents:\n{doc_output}"

        if not answer:
            answer = "No relevant internal documents found."

        return {
            "messages": [HumanMessage(content=f"[RAG Specialist]: {answer}", name="rag_agent")],
            "agent_outputs": [
                AgentOutput(agent_name="rag_agent", result=answer, metadata={"context": context})
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4. SQL Sub-Graph Node (Structured Database Specialist)
# ─────────────────────────────────────────────────────────────────────────────
class SQLSubGraphNode:
    """Specialized node executing read-only SQL queries and schema inspections."""

    def __init__(self, llm: ClaudeService, tool_registry: ToolRegistry):
        self.llm = llm
        self.registry = tool_registry

    async def __call__(self, state: SupervisorState) -> Dict[str, Any]:
        logger.info("[SQL Node] Executing database query specialist...")
        query = state.get("query", "")

        schema_tool = self.registry.get_tool("sql_db_schema")
        query_tool = self.registry.get_tool("sql_db_query")

        if not schema_tool or not query_tool:
            err = "SQL tools not configured in tool registry."
            return {
                "agent_outputs": [AgentOutput(agent_name="sql_agent", result=err)],
            }

        try:
            # 1. Fetch schema
            schema_info = await schema_tool.ainvoke({})

            # 2. Generate read-only SQL query
            sql_prompt = f"""\
You are an expert SQL analyst. Given the following database schema and user question, write a single valid SQLite SELECT query.
Output ONLY the raw SQL query with no markdown blocks or surrounding text.

Schema:
{schema_info}

User Question: {query}"""

            generated_sql = await self.llm.agenerate(prompt=sql_prompt, max_tokens=256)
            cleaned_sql = generated_sql.replace("```sql", "").replace("```", "").strip()

            # 3. Execute SQL query
            query_result = await query_tool.ainvoke({"query": cleaned_sql})

            # 4. Summarize result
            summary_prompt = f"""\
User Question: {query}
Executed SQL: {cleaned_sql}
Database Result:
{query_result}

Provide a clear, factual summary of the database query results:"""

            summary = await self.llm.agenerate(prompt=summary_prompt, max_tokens=512)
            final_sql_result = f"Query: `{cleaned_sql}`\n\nResult:\n{summary}"

        except Exception as e:
            logger.error(f"[SQL Node] SQL execution failed: {e}", exc_info=True)
            final_sql_result = f"Database query could not be completed: {str(e)}"

        return {
            "messages": [HumanMessage(content=f"[SQL Specialist]: {final_sql_result}", name="sql_agent")],
            "agent_outputs": [
                AgentOutput(agent_name="sql_agent", result=final_sql_result)
            ],
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Web Agent Node (Online Search Specialist)
# ─────────────────────────────────────────────────────────────────────────────
class WebAgentNode:
    """Specialized node performing live internet search and research synthesis."""

    def __init__(self, llm: ClaudeService, tool_registry: ToolRegistry):
        self.llm = llm
        self.registry = tool_registry

    async def __call__(self, state: SupervisorState) -> Dict[str, Any]:
        logger.info("[Web Agent Node] Executing live web search...")
        query = state.get("query", "")
        search_tool = self.registry.get_tool("web_search")

        if not search_tool:
            return {
                "agent_outputs": [
                    AgentOutput(agent_name="web_agent", result="Web search tool is not available.")
                ],
            }

        try:
            search_raw = await search_tool.ainvoke({"query": query, "max_results": 5})
            synth_prompt = f"""\
You already have live web search results. Use only those results.
Never say you cannot browse the internet or that your knowledge is outdated.

Question: {query}

Search Results:
{search_raw}

Answer with the latest figure or fact from the results, and mention the source if a URL is present:"""
            answer = await self.llm.agenerate(prompt=synth_prompt, max_tokens=600)
        except Exception as e:
            logger.error(f"[Web Agent] Search error: {e}", exc_info=True)
            answer = f"Web search could not retrieve information: {str(e)}"

        return {
            "messages": [HumanMessage(content=f"[Web Specialist]: {answer}", name="web_agent")],
            "agent_outputs": [AgentOutput(agent_name="web_agent", result=answer)],
        }


# ─────────────────────────────────────────────────────────────────────────────
# 6. General LLM Node (Conversational & Direct Knowledge)
# ─────────────────────────────────────────────────────────────────────────────
class GeneralLLMNode:
    """Specialist node for general knowledge, coding, and chitchat."""

    def __init__(self, llm: ClaudeService):
        self.llm = llm

    async def __call__(self, state: SupervisorState) -> Dict[str, Any]:
        logger.info("[General Node] Executing direct LLM reasoning...")
        query = state.get("query", "")
        memories = state.get("long_term_memories") or []
        memories_str = "\n".join(f"- {m}" for m in memories) or "(none)"

        system_prompt = f"""\
You are a knowledgeable, direct AI assistant.
User Profile Facts:
{memories_str}

Answer the user's conversational or general reasoning question directly."""

        response = await self.llm.agenerate(
            prompt=f"User Query: {query}",
            system_prompt=system_prompt,
            max_tokens=1024,
        )

        return {
            "messages": [HumanMessage(content=f"[General Specialist]: {response}", name="general_agent")],
            "agent_outputs": [AgentOutput(agent_name="general_agent", result=response)],
        }


# ─────────────────────────────────────────────────────────────────────────────
# 7. Output Guardrail Node (Exit-Door Protection)
# ─────────────────────────────────────────────────────────────────────────────
class OutputGuardrailNode:
    """Validates the Supervisor's synthesized response before delivery."""

    def __init__(self, guardrail_service: OutputGuardrailService | None = None):
        self.guardrail_service = guardrail_service

    async def __call__(self, state: SupervisorState) -> Dict[str, Any]:
        if self.guardrail_service is None:
            return {
                "output_guardrail_passed": True,
                "output_guardrail_reason": None,
            }

        query = state.get("query", "")
        final_response = state.get("final_response", "")
        
        # Combine context from all agent outputs
        context_snippets = [
            f"[{o.agent_name}]: {o.result}" for o in state.get("agent_outputs", [])
        ]
        full_context = "\n\n".join(context_snippets)

        logger.info("[OutputGuardrail] Validating final response at supervisor boundary...")
        result = await self.guardrail_service.validate(
            query=query,
            response=final_response,
            context=full_context,
        )

        sanitized_response = final_response
        # Check if PII guardrail anonymized the text
        if result.metadata.get("anonymized_response"):
            sanitized_response = result.metadata["anonymized_response"]
            logger.info("[OutputGuardrail] PII redacted from final response.")

        if not result.passed and result.action == "block":
            logger.warning(f"[OutputGuardrail] Final response BLOCKED: {result.reason}")

        return {
            "output_guardrail_passed": result.passed or result.action == "anonymize",
            "output_guardrail_reason": result.reason,
            "final_response": sanitized_response,
            "guardrail_metadata": {
                **state.get("guardrail_metadata", {}),
                **result.metadata,
                "output_action": result.action,
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# 8. Save Memory Node (Supervisor-Level Long-Term Memory Extraction)
# ─────────────────────────────────────────────────────────────────────────────
class SaveMemoryNode:
    """Extracts durable user facts from the complete turn and persists them to SQLite."""

    def __init__(self, llm: ClaudeService):
        self.llm = llm

    async def __call__(self, state: SupervisorState) -> Dict[str, Any]:
        user_id = state.get("user_id")
        query = state.get("query", "")
        final_response = state.get("final_response", "")

        persist_from_turn_background(self.llm, user_id, query, final_response)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# 9. Error Handler Node
# ─────────────────────────────────────────────────────────────────────────────
class ErrorHandlerNode:
    """Returns safe failure or guardrail block message to user."""

    async def __call__(self, state: SupervisorState) -> Dict[str, Any]:
        reason = (
            state.get("input_guardrail_reason")
            or state.get("output_guardrail_reason")
            or state.get("error")
            or "Request could not be processed due to a safety policy."
        )
        safe_response = f"I cannot complete your request. Reason: {reason}"
        logger.info(f"[ErrorHandler] Returning blocked/error response: {safe_response}")

        return {
            "final_response": safe_response,
            "messages": [AIMessage(content=safe_response)],
        }