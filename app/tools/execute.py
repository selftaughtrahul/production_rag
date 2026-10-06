"""Invoke a registry tool with audit logging and human-approval gate."""

from __future__ import annotations

from typing import Any

from app.core.audit import audit_event

_TENANT_SCOPED_TOOLS = frozenset(
    {
        "document_search",
        "list_user_documents",
    }
)


async def invoke_tool(
    tool: Any,
    payload: dict[str, Any],
    *,
    user_id: str | None = None,
    session_id: str | None = None,
    approved: bool = False,
    doc_tags: list[str] | None = None,
) -> Any:
    """Run a tool, or refuse write/high-risk tools until approved.

    ``user_id`` and ``doc_tags`` are injected by the orchestrator, never by
    the model, so the tenant and tag filters cannot be dropped.
    """
    name = getattr(tool, "name", "unknown")
    safe_payload = dict(payload)
    if name in _TENANT_SCOPED_TOOLS:
        if not user_id:
            raise PermissionError(f"Tool '{name}' requires an authenticated user.")
        safe_payload["user_id"] = user_id
    if name == "document_search":
        safe_payload["tags"] = list(doc_tags or [])
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
    try:
        result = await tool.ainvoke(safe_payload)
    except Exception:
        audit_event(
            "tool.error",
            user_id=user_id,
            session_id=session_id,
            data={"tool": name},
        )
        raise
    audit_event(
        "tool.done",
        user_id=user_id,
        session_id=session_id,
        data={"tool": name, "success": True},
    )
    return result
