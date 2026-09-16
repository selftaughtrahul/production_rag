"""Database and order specialist with checkpoint-backed write confirmation."""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Literal
from uuid import uuid4

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, ValidationError

from app.agent.multi_agent.state import AgentOutput, PendingOrderAction, SupervisorState
from app.core.audit import audit_event
from app.prompts.specialist_prompts import (
    ORDER_INTENT_SYSTEM_V1,
)
from app.services.llm.claude import ClaudeService
from app.tools.execute import invoke_tool
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_CONFIRM_RE = re.compile(r"^confirm order (act_[a-f0-9]{8})$", re.IGNORECASE)
_CANCEL_RE = re.compile(r"^cancel order (act_[a-f0-9]{8})$", re.IGNORECASE)
_ORDER_ID_RE = re.compile(r"\bord_[a-zA-Z0-9_-]+\b")
_DAYS_RE = re.compile(r"\b(?:last|past)\s+(\d{1,3})\s+days?\b", re.IGNORECASE)


class OrderItemDraft(BaseModel):
    sku: str | None = None
    name: str | None = None
    quantity: int | None = Field(default=None, ge=1, le=100)
    unit_price_cents: int | None = Field(default=None, ge=0)


class OrderIntentV1(BaseModel):
    action: Literal[
        "create_order",
        "update_order",
        "get_order",
        "list_orders",
        "order_analytics",
        "other",
    ]
    order_id: str | None = None
    days: int = Field(default=10, ge=1, le=365)
    status: str | None = None
    items: list[OrderItemDraft] = Field(default_factory=list)
    shipping_address: str | None = None
    currency: str | None = None
    note: str | None = None


class SQLSubGraphNode:
    def __init__(self, llm: ClaudeService, tool_registry: ToolRegistry) -> None:
        self.llm = llm
        self.registry = tool_registry

    async def __call__(self, state: SupervisorState) -> dict[str, Any]:
        pending_raw = state.get("pending_order_action")
        if pending_raw:
            return await self._continue_pending(state, pending_raw)
        query = state.get("query", "")
        if "order" in query.lower() or _ORDER_ID_RE.search(query):
            return await self._handle_order_intent(state)
        return await self._run_generic_sql(state)

    async def _handle_order_intent(self, state: SupervisorState) -> dict[str, Any]:
        query = state.get("query", "")
        lowered = query.lower()
        user_id = state.get("user_id", "")
        session_id = state.get("thread_id", "")
        days_match = _DAYS_RE.search(query)
        days = int(days_match.group(1)) if days_match else 10

        if any(word in lowered for word in ("analytics", "summary", "total", "count")):
            return await self._invoke_read(
                state,
                "order_analytics",
                {"user_id": user_id, "days": days},
            )
        if any(word in lowered for word in ("last", "recent", "list", "show")):
            return await self._invoke_read(
                state,
                "list_orders",
                {"user_id": user_id, "days": days, "limit": 50},
            )
        order_id = _ORDER_ID_RE.search(query)
        if order_id and not any(word in lowered for word in ("update", "change", "cancel")):
            return await self._invoke_read(
                state,
                "get_order",
                {"user_id": user_id, "order_id": order_id.group(0)},
            )

        intent = await self._extract_intent(query, draft=None)
        if intent is None:
            return self._result(
                "I could not reliably parse that order request. Please restate it "
                "with only the intended order operation and details.",
                direct=True,
            )
        if intent.action not in {"create_order", "update_order"}:
            return self._result(
                "I can get, list, analyze, create, or update orders. "
                "Please include the order operation you want.",
                direct=True,
            )
        return await self._start_pending(intent, user_id=user_id, session_id=session_id)

    async def _start_pending(
        self,
        intent: OrderIntentV1,
        *,
        user_id: str,
        session_id: str,
    ) -> dict[str, Any]:
        tool_name = intent.action
        payload = self._intent_payload(intent)
        if tool_name == "update_order":
            await self._fill_current_version(payload, user_id=user_id, session_id=session_id)
        missing = self._missing_fields(tool_name, payload)
        action_id = f"act_{uuid4().hex[:8]}"
        pending = PendingOrderAction(
            action_id=action_id,
            user_id=user_id,
            session_id=session_id,
            tool_name=tool_name,
            payload=payload,
            missing_fields=missing,
            phase="collecting" if missing else "awaiting_confirmation",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
            idempotency_key=f"{session_id}:{action_id}",
        )
        self._seal_if_complete(pending)
        audit_event(
            "order.draft_created",
            user_id=user_id,
            session_id=session_id,
            data={"action_id": action_id, "tool": tool_name, "missing": missing},
        )
        return self._pending_response(pending)

    async def _continue_pending(
        self,
        state: SupervisorState,
        pending_raw: dict[str, Any],
    ) -> dict[str, Any]:
        query = state.get("query", "").strip()
        user_id = state.get("user_id", "")
        session_id = state.get("thread_id", "")
        try:
            pending = PendingOrderAction.model_validate(pending_raw)
        except ValueError:
            return self._result(
                "The pending order draft is invalid and was cleared.",
                pending=None,
                direct=True,
            )
        if pending.user_id != user_id or pending.session_id != session_id:
            audit_event(
                "order.confirmation_denied",
                user_id=user_id,
                session_id=session_id,
                data={"reason": "owner_or_session_mismatch"},
            )
            return self._result(
                "That pending order action does not belong to this session.",
                pending=None,
                direct=True,
            )
        if pending.expires_at <= datetime.now(timezone.utc):
            audit_event(
                "order.draft_expired",
                user_id=user_id,
                session_id=session_id,
                data={"action_id": pending.action_id},
            )
            return self._result(
                "The pending order action expired. Please start again.",
                pending=None,
                direct=True,
            )

        cancel_match = _CANCEL_RE.fullmatch(query)
        if cancel_match:
            if cancel_match.group(1).lower() != pending.action_id:
                return self._mismatched_action(pending)
            audit_event(
                "order.draft_cancelled",
                user_id=user_id,
                session_id=session_id,
                data={"action_id": pending.action_id},
            )
            return self._result("Order action cancelled.", pending=None, direct=True)

        confirm_match = _CONFIRM_RE.fullmatch(query)
        if confirm_match:
            if confirm_match.group(1).lower() != pending.action_id:
                return self._mismatched_action(pending)
            if pending.phase != "awaiting_confirmation" or pending.missing_fields:
                return self._pending_response(pending)
            return await self._execute_pending(pending)

        if pending.phase == "awaiting_confirmation":
            return self._pending_response(pending)

        intent = await self._extract_intent(query, draft=pending.payload)
        if intent is None:
            return self._result(
                "I could not parse those details. The draft was not changed; "
                "please provide the requested fields again.",
                pending=pending.model_dump(mode="json"),
                direct=True,
            )
        updates = self._intent_payload(intent)
        pending.payload.update(
            {key: value for key, value in updates.items() if value not in (None, [], "")}
        )
        if pending.tool_name == "update_order":
            await self._fill_current_version(
                pending.payload,
                user_id=user_id,
                session_id=session_id,
            )
        pending.missing_fields = self._missing_fields(
            pending.tool_name,
            pending.payload,
        )
        pending.phase = (
            "collecting" if pending.missing_fields else "awaiting_confirmation"
        )
        self._seal_if_complete(pending)
        return self._pending_response(pending)

    async def _execute_pending(
        self,
        pending: PendingOrderAction,
    ) -> dict[str, Any]:
        tool = self.registry.get_tool_for_agent("sql_agent", pending.tool_name)
        if not tool:
            return self._result(
                "The order write tool is unavailable.",
                pending=pending.model_dump(mode="json"),
                direct=True,
            )
        if (
            pending.payload_hash is None
            or pending.payload_hash != self._payload_hash(pending.payload)
        ):
            return self._result(
                "The order draft changed after review and was not executed.",
                pending=None,
                direct=True,
            )
        payload = dict(pending.payload)
        payload["user_id"] = pending.user_id
        if pending.tool_name == "create_order":
            payload["idempotency_key"] = pending.idempotency_key
        output = await invoke_tool(
            tool,
            payload,
            user_id=pending.user_id,
            session_id=pending.session_id,
            approved=True,
        )
        audit_event(
            "order.write_executed",
            user_id=pending.user_id,
            session_id=pending.session_id,
            data={"action_id": pending.action_id, "tool": pending.tool_name},
        )
        return self._result(
            f"Confirmed and executed {pending.tool_name}.\n\n{output}",
            pending=None,
            direct=True,
        )

    async def _fill_current_version(
        self,
        payload: dict[str, Any],
        *,
        user_id: str,
        session_id: str,
    ) -> None:
        if payload.get("expected_version") or not payload.get("order_id"):
            return
        tool = self.registry.get_tool_for_agent("sql_agent", "get_order")
        if not tool:
            return
        raw = await invoke_tool(
            tool,
            {"user_id": user_id, "order_id": payload["order_id"]},
            user_id=user_id,
            session_id=session_id,
        )
        try:
            parsed = ast.literal_eval(str(raw))
        except (SyntaxError, ValueError):
            return
        if isinstance(parsed, dict) and isinstance(parsed.get("version"), int):
            payload["expected_version"] = parsed["version"]

    async def _invoke_read(
        self,
        state: SupervisorState,
        tool_name: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        tool = self.registry.get_tool_for_agent("sql_agent", tool_name)
        if not tool:
            return self._result(f"{tool_name} is not configured.", direct=True)
        output = await invoke_tool(
            tool,
            payload,
            user_id=state.get("user_id"),
            session_id=state.get("thread_id"),
        )
        return self._result(str(output), direct=True, operational=True)

    async def _extract_intent(
        self,
        query: str,
        *,
        draft: dict[str, Any] | None,
    ) -> OrderIntentV1 | None:
        try:
            return await self.llm.agenerate_structured(
                prompt=(
                    f"Existing draft data:\n{json.dumps(draft or {}, default=str)}\n\n"
                    f"User message:\n{query}"
                ),
                system_prompt=ORDER_INTENT_SYSTEM_V1,
                response_model=OrderIntentV1,
            )
        except ValidationError:
            logger.warning("Order intent response failed schema validation")
            return None

    async def _run_generic_sql(self, state: SupervisorState) -> dict[str, Any]:
        return self._result(
            "Direct queries against application tables are disabled. "
            "I can provide tenant-scoped order reads and analytics.",
            direct=True,
        )

    @staticmethod
    def _intent_payload(intent: OrderIntentV1) -> dict[str, Any]:
        data = intent.model_dump(exclude_none=True)
        data.pop("action", None)
        data.pop("days", None)
        if not data.get("items"):
            data.pop("items", None)
        data.setdefault("currency", "USD")
        return data

    @staticmethod
    def _missing_fields(tool_name: str, payload: dict[str, Any]) -> list[str]:
        if tool_name == "create_order":
            missing: list[str] = []
            if not payload.get("shipping_address"):
                missing.append("shipping_address")
            items = payload.get("items") or []
            if not items:
                missing.append("items")
            else:
                for index, item in enumerate(items, 1):
                    for field in ("sku", "name", "quantity", "unit_price_cents"):
                        if item.get(field) in (None, ""):
                            missing.append(f"items[{index}].{field}")
            return missing
        missing = []
        if not payload.get("order_id"):
            missing.append("order_id")
        if not payload.get("expected_version"):
            missing.append("current_order_version")
        if not any(payload.get(key) is not None for key in ("status", "shipping_address", "note")):
            missing.append("field_to_update")
        return missing

    @staticmethod
    def _payload_hash(payload: dict[str, Any]) -> str:
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def _seal_if_complete(cls, pending: PendingOrderAction) -> None:
        if pending.missing_fields:
            pending.payload_hash = None
            return
        pending.payload_hash = cls._payload_hash(pending.payload)
        pending.idempotency_key = (
            f"{pending.session_id}:{pending.action_id}:{pending.payload_hash[:16]}"
        )

    @classmethod
    def _pending_response(cls, pending: PendingOrderAction) -> dict[str, Any]:
        if pending.missing_fields:
            text = (
                "I need the following information before preparing the order action: "
                + ", ".join(pending.missing_fields)
                + "."
            )
            audit_name = "details_required"
        else:
            summary = cls._summary(pending)
            text = (
                f"Please review this order action:\n{summary}\n\n"
                f"Type exactly `confirm order {pending.action_id}` to execute it, "
                f"or `cancel order {pending.action_id}`."
            )
            audit_name = "confirmation_required"
        audit_event(
            f"order.{audit_name}",
            user_id=pending.user_id,
            session_id=pending.session_id,
            data={
                "action_id": pending.action_id,
                "tool": pending.tool_name,
                "missing": pending.missing_fields,
            },
        )
        return cls._result(
            text,
            pending=pending.model_dump(mode="json"),
            direct=True,
        )

    @staticmethod
    def _summary(pending: PendingOrderAction) -> str:
        payload = pending.payload
        if pending.tool_name == "create_order":
            items = payload.get("items") or []
            item_lines = ", ".join(
                f"{item.get('quantity')} × {item.get('name')} ({item.get('sku')})"
                for item in items
            )
            total = sum(
                int(item.get("quantity", 0)) * int(item.get("unit_price_cents", 0))
                for item in items
            )
            return (
                f"Create order: {item_lines}; total {total / 100:.2f} "
                f"{payload.get('currency', 'USD')}; ship to {payload.get('shipping_address')}."
            )
        changes = {
            key: payload.get(key)
            for key in ("status", "shipping_address", "note")
            if payload.get(key) is not None
        }
        return f"Update order {payload.get('order_id')} with {changes}."

    @classmethod
    def _mismatched_action(cls, pending: PendingOrderAction) -> dict[str, Any]:
        return cls._result(
            f"That action ID does not match the pending action {pending.action_id}.",
            pending=pending.model_dump(mode="json"),
            direct=True,
        )

    @staticmethod
    def _result(
        answer: str,
        *,
        pending: dict[str, Any] | None | object = ...,
        direct: bool = False,
        operational: bool = True,
    ) -> dict[str, Any]:
        metadata = {
            "direct_response": direct,
            "operational": operational,
        }
        update: dict[str, Any] = {
            "messages": [
                HumanMessage(content=f"[SQL Specialist]: {answer}", name="sql_agent")
            ],
            "agent_outputs": [
                AgentOutput(agent_name="sql_agent", result=answer, metadata=metadata)
            ],
        }
        if pending is not ...:
            update["pending_order_action"] = pending
        return update
