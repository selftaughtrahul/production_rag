"""Invoke a registry tool with audit logging and human-approval gate."""

from __future__ import annotations

from typing import Any

from app.core.audit import audit_event


async def invoke_tool(
    tool: Any,
    payload: dict[str, Any],
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    approved: bool = False,
) -> Any:
    """Run a tool, or refuse write/high-risk tools until approved."""
    name = getattr(tool, "name", "unknown")
    if getattr(tool, "requires_approval", False) and not approved:
        audit_event(
            "tool.denied",
            user_id=user_id,
            session_id=session_id,
            data={"tool": name, "reason": "approval_required"},
        )
        return "This tool requires human approval and was not executed."
    audit_event(
        "tool.call",
        user_id=user_id,
        session_id=session_id,
        data={"tool": name},
    )
    return await tool.ainvoke(payload)
