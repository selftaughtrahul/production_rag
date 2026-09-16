"""Conversational order draft and confirmation state transitions."""

from __future__ import annotations

import asyncio
import ast
from datetime import datetime, timedelta, timezone

from app.agent.multi_agent.specialists.sql import (
    OrderIntentV1,
    OrderItemDraft,
    SQLSubGraphNode,
)
from app.agent.multi_agent.master_graph import build_master_agent_graph
from app.tools.order_tools import (
    CreateOrderTool,
    GetOrderTool,
    ListOrdersTool,
    OrderAnalyticsTool,
    UpdateOrderTool,
)
from app.tools.registry import ToolRegistry
from database.order_store import OrderRepository, create_order_engine, init_order_db
from database.seed_orders import seed_demo_orders


class FakeOrderLLM:
    async def agenerate_structured(self, *, prompt, system_prompt, response_model):
        message = prompt.split("User message:\n", 1)[-1].lower()
        if "update order" in message:
            order_id = message.split("update order ", 1)[1].split()[0]
            return OrderIntentV1(
                action="update_order",
                order_id=order_id,
                status="shipped",
            )
        if "ship it to" in message:
            return OrderIntentV1(
                action="create_order",
                shipping_address="1 Main Street, Test City",
            )
        if message.strip() == "yes":
            return OrderIntentV1(action="other")
        return OrderIntentV1(
            action="create_order",
            items=[
                OrderItemDraft(
                    sku="SKU-1",
                    name="Widget",
                    quantity=2,
                    unit_price_cents=500,
                )
            ],
        )


def _node(tmp_path):
    engine = create_order_engine(f"sqlite:///{tmp_path / 'orders.db'}")
    init_order_db(engine)
    repository = OrderRepository(engine)
    registry = ToolRegistry()
    for tool in (
        GetOrderTool(repository=repository),
        ListOrdersTool(repository=repository),
        OrderAnalyticsTool(repository=repository),
        CreateOrderTool(repository=repository),
        UpdateOrderTool(repository=repository),
    ):
        registry.register_tool(tool)
    return SQLSubGraphNode(FakeOrderLLM(), registry), repository


def _registry_and_repository(tmp_path):
    engine = create_order_engine(f"sqlite:///{tmp_path / 'graph-orders.db'}")
    init_order_db(engine)
    repository = OrderRepository(engine)
    registry = ToolRegistry()
    for tool in (
        GetOrderTool(repository=repository),
        ListOrdersTool(repository=repository),
        OrderAnalyticsTool(repository=repository),
        CreateOrderTool(repository=repository),
        UpdateOrderTool(repository=repository),
    ):
        registry.register_tool(tool)
    return registry, repository


def _state(query: str, pending=None):
    state = {
        "query": query,
        "user_id": "user-a",
        "thread_id": "session-test",
    }
    if pending is not None:
        state["pending_order_action"] = pending
    return state


def test_incomplete_details_then_exact_confirmation_executes(tmp_path) -> None:
    node, repository = _node(tmp_path)

    first = asyncio.run(node(_state("create order with two widgets")))
    pending = first["pending_order_action"]
    assert pending["phase"] == "collecting"
    assert "shipping_address" in pending["missing_fields"]

    second = asyncio.run(
        node(_state("ship it to 1 Main Street, Test City", pending))
    )
    pending = second["pending_order_action"]
    assert pending["phase"] == "awaiting_confirmation"
    assert pending["missing_fields"] == []

    plain_yes = asyncio.run(node(_state("yes", pending)))
    assert plain_yes["pending_order_action"]["phase"] == "awaiting_confirmation"
    assert repository.list_orders(user_id="user-a") == []

    confirmed = asyncio.run(
        node(_state(f"confirm order {pending['action_id']}", pending))
    )
    assert confirmed["pending_order_action"] is None
    orders = repository.list_orders(user_id="user-a")
    assert len(orders) == 1
    assert orders[0]["total_cents"] == 1000


def test_cancel_and_mismatched_action_do_not_execute(tmp_path) -> None:
    node, repository = _node(tmp_path)
    first = asyncio.run(node(_state("create order with two widgets")))
    second = asyncio.run(
        node(
            _state(
                "ship it to 1 Main Street, Test City",
                first["pending_order_action"],
            )
        )
    )
    pending = second["pending_order_action"]
    mismatch = asyncio.run(node(_state("confirm order act_deadbeef", pending)))
    assert mismatch["pending_order_action"]["action_id"] == pending["action_id"]

    cancelled = asyncio.run(
        node(_state(f"cancel order {pending['action_id']}", pending))
    )
    assert cancelled["pending_order_action"] is None
    assert repository.list_orders(user_id="user-a") == []


def test_expired_or_wrong_tenant_pending_action_is_cleared(tmp_path) -> None:
    node, repository = _node(tmp_path)
    first = asyncio.run(node(_state("create order with two widgets")))
    pending = first["pending_order_action"]
    pending["expires_at"] = (
        datetime.now(timezone.utc) - timedelta(seconds=1)
    ).isoformat()
    expired = asyncio.run(node(_state(f"confirm order {pending['action_id']}", pending)))
    assert expired["pending_order_action"] is None

    fresh = asyncio.run(node(_state("create order with two widgets")))
    other_user_state = {
        "query": f"confirm order {fresh['pending_order_action']['action_id']}",
        "user_id": "user-b",
        "thread_id": "session-test",
        "pending_order_action": fresh["pending_order_action"],
    }
    denied = asyncio.run(node(other_user_state))
    assert denied["pending_order_action"] is None
    assert repository.list_orders(user_id="user-a") == []
    assert repository.list_orders(user_id="user-b") == []


def test_pending_action_survives_master_graph_checkpoint(tmp_path) -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    registry, repository = _registry_and_repository(tmp_path)
    graph = build_master_agent_graph(
        llm=FakeOrderLLM(),
        tool_registry=registry,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "session-graph"}}

    async def run_turns():
        common = {
            "user_id": "user-a",
            "thread_id": "session-graph",
            "iterations": 0,
            "max_iterations": 2,
        }
        first = await graph.ainvoke(
            {**common, "query": "create order with two widgets"},
            config=config,
        )
        second = await graph.ainvoke(
            {**common, "query": "ship it to 1 Main Street, Test City"},
            config=config,
        )
        pending = second["pending_order_action"]
        third = await graph.ainvoke(
            {**common, "query": f"confirm order {pending['action_id']}"},
            config=config,
        )
        return first, second, third

    first, second, third = asyncio.run(run_turns())
    assert first["pending_order_action"]["phase"] == "collecting"
    assert second["pending_order_action"]["phase"] == "awaiting_confirmation"
    assert third["pending_order_action"] is None
    assert len(repository.list_orders(user_id="user-a")) == 1


def test_update_order_requires_confirmation_and_uses_current_version(tmp_path) -> None:
    node, repository = _node(tmp_path)
    order = repository.create_order(
        user_id="user-a",
        items=[
            {
                "sku": "SKU-1",
                "name": "Widget",
                "quantity": 1,
                "unit_price_cents": 500,
            }
        ],
        shipping_address="1 Main Street",
        currency="USD",
        note=None,
        idempotency_key="update-fixture",
    )
    draft = asyncio.run(node(_state(f"update order {order['id']} to shipped")))
    pending = draft["pending_order_action"]
    assert pending["phase"] == "awaiting_confirmation"
    assert pending["payload"]["expected_version"] == 1
    assert repository.get_order(user_id="user-a", order_id=order["id"])["status"] == "pending"

    asyncio.run(node(_state(f"confirm order {pending['action_id']}", pending)))
    assert repository.get_order(user_id="user-a", order_id=order["id"])["status"] == "shipped"


def test_specialist_lists_last_ten_days_and_rejects_payload_tamper(tmp_path) -> None:
    node, repository = _node(tmp_path)
    seed_demo_orders(repository, user_id="user-a", days=30)
    listed = asyncio.run(node(_state("show my orders from the last 10 days")))
    listed_orders = ast.literal_eval(listed["agent_outputs"][0].result)
    assert 9 <= len(listed_orders) <= 11
    assert all(order["user_id"] == "user-a" for order in listed_orders)

    first = asyncio.run(node(_state("create order with two widgets")))
    second = asyncio.run(
        node(
            _state(
                "ship it to 1 Main Street, Test City",
                first["pending_order_action"],
            )
        )
    )
    pending = second["pending_order_action"]
    pending["payload"]["shipping_address"] = "Changed after review"
    denied = asyncio.run(
        node(_state(f"confirm order {pending['action_id']}", pending))
    )
    assert denied["pending_order_action"] is None
    assert repository.list_orders(user_id="user-a", days=365)[0]["id"].startswith("ord_")
    assert len(repository.list_orders(user_id="user-a", days=365)) == 30
