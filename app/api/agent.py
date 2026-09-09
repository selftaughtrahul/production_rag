import logging
import uuid
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_master_agent_graph
from app.models.schemas import AgentQueryRequest, AgentQueryResponse, UserInDB
from app.services.auth.dependencies import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["Multi-Agent Orchestrator"])


def _generate_session_id() -> str:
    return f"session_agent_{uuid.uuid4()}"


@router.post(
    "/run",
    response_model=AgentQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute multi-agent orchestration with supervisor guardrails & memory",
)
async def run_multi_agent(
    request: AgentQueryRequest,
    master_graph=Depends(get_master_agent_graph),
    current_user: UserInDB = Depends(get_current_user),
):
    """
    Executes a user question across the specialized multi-agent system:
    - Supervisor-level front-door Input Guardrails (Injection, PII, Safety)
    - Supervisor-level memory (Long-term facts & LangGraph thread checkpointer)
    - Dynamic routing to RAG, SQL, Web, or General LLM specialists
    - Supervisor-level exit-door Output Guardrails (Safety, Grounding, PII redaction)
    - Automatic turn memory extraction & persistence
    """
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty",
        )

    thread_id = request.session_id or _generate_session_id()

    input_state = {
        "query": request.query,
        "user_id": current_user.id,
        "thread_id": thread_id,
        "iterations": 0,
        "max_iterations": 5,
    }

    config = {
        "configurable": {"thread_id": thread_id},
        "run_name": f"multi-agent-turn-{thread_id}",
        "metadata": {
            "user_id": current_user.id,
            "session_id": thread_id,
            "source": "multi_agent_api",
        },
    }

    try:
        logger.info(f"Executing multi-agent turn for user='{current_user.id}', session='{thread_id}'")
        result = await master_graph.ainvoke(input_state, config=config)

        # Extract agent trajectory for full visibility
        trajectory = []
        for out in result.get("agent_outputs", []):
            trajectory.append({
                "agent_name": out.agent_name,
                "result": out.result,
                "metadata": out.metadata,
            })

        final_answer = result.get("final_response") or "Unable to synthesize answer."

        return AgentQueryResponse(
            success=True,
            session_id=thread_id,
            query=request.query,
            final_answer=final_answer,
            iterations=result.get("iterations", 1),
            agent_trajectory=trajectory,
            guardrail_metadata=result.get("guardrail_metadata", {}),
        )

    except Exception as exc:
        logger.error(f"Multi-agent execution failure: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution encountered an error: {str(exc)}",
        )
