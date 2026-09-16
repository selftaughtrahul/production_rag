"""Isolated order repository behavior."""

from __future__ import annotations

import pytest

from database.order_store import OrderRepository, create_order_engine, init_order_db
from database.seed_orders import seed_demo_orders


def _repository(tmp_path) -> OrderRepository:
    engine = create_order_engine(f"sqlite:///{tmp_path / 'orders.db'}")
    init_order_db(engine)
    return OrderRepository(engine)


def test_seed_and_last_ten_days_are_tenant_scoped(tmp_path) -> None:
    repository = _repository(tmp_path)
    seed_demo_orders(repository, user_id="user-a", days=30)
    seed_demo_orders(repository, user_id="user-b", days=30)

    recent = repository.list_orders(user_id="user-a", days=10)

    assert 9 <= len(recent) <= 11
    assert all(order["user_id"] == "user-a" for order in recent)
    assert repository.analytics(user_id="user-a", days=10)["orders"] == len(recent)


def test_create_is_idempotent_and_update_uses_version(tmp_path) -> None:
    repository = _repository(tmp_path)
    payload = {
        "user_id": "user-a",
        "items": [
            {
                "sku": "SKU-1",
                "name": "Widget",
                "quantity": 2,
                "unit_price_cents": 500,
            }
        ],
        "shipping_address": "1 Main Street",
        "currency": "USD",
        "note": None,
        "idempotency_key": "action-1",
    }
    first = repository.create_order(**payload)
    repeated = repository.create_order(**payload)
    assert repeated["id"] == first["id"]
    assert first["total_cents"] == 1000
    changed_payload = {
        **payload,
        "shipping_address": "2 Different Street",
    }
    with pytest.raises(ValueError, match="different order"):
        repository.create_order(**changed_payload)

    updated = repository.update_order(
        user_id="user-a",
        order_id=first["id"],
        expected_version=1,
        status="confirmed",
    )
    assert updated is not None
    assert updated["status"] == "confirmed"
    assert updated["version"] == 2
    assert (
        repository.update_order(
            user_id="user-a",
            order_id=first["id"],
            expected_version=1,
            status="cancelled",
        )
        is None
    )


def test_get_order_cannot_cross_tenants(tmp_path) -> None:
    repository = _repository(tmp_path)
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
        idempotency_key="tenant-test",
    )
    assert repository.get_order(user_id="user-a", order_id=order["id"]) is not None
    assert repository.get_order(user_id="user-b", order_id=order["id"]) is None


def test_seed_is_deterministic_across_fresh_databases(tmp_path) -> None:
    first = _repository(tmp_path / "first")
    second = _repository(tmp_path / "second")
    seed_demo_orders(first, user_id="user-a", days=5)
    seed_demo_orders(second, user_id="user-a", days=5)

    def stable_fields(repository: OrderRepository):
        return [
            (order["id"], order["status"], order["total_cents"])
            for order in repository.list_orders(user_id="user-a", days=10)
        ]

    assert stable_fields(first) == stable_fields(second)
