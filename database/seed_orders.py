"""Seed deterministic demo orders for one authenticated user.

Usage:
    python -m database.seed_orders --user-id <user-id>
"""

from __future__ import annotations

import argparse
from datetime import datetime, time, timedelta, timezone
from uuid import NAMESPACE_URL, uuid5

from app.core.config import Settings
from database.order_store import OrderRepository, create_order_engine, init_order_db

_PRODUCTS = (
    {"sku": "SKU-TSHIRT", "name": "RAG T-Shirt", "unit_price_cents": 2500},
    {"sku": "SKU-MUG", "name": "Vector Search Mug", "unit_price_cents": 1600},
    {"sku": "SKU-NOTE", "name": "Agent Notebook", "unit_price_cents": 1200},
)
_STATUSES = ("pending", "confirmed", "processing", "shipped", "cancelled")


def seed_demo_orders(
    repository: OrderRepository,
    *,
    user_id: str,
    days: int = 30,
) -> int:
    """Create one deterministic idempotent demo order per day."""
    now = datetime.combine(
        datetime.now(timezone.utc).date(),
        time.min,
    )
    for offset in range(days):
        product = _PRODUCTS[offset % len(_PRODUCTS)]
        order = repository.create_order(
            user_id=user_id,
            items=[{**product, "quantity": (offset % 3) + 1}],
            shipping_address="100 Demo Street, Test City",
            currency="USD",
            note=f"Demo order day {offset + 1}",
            idempotency_key=f"demo:{user_id}:{offset}",
            created_at=now - timedelta(days=offset),
            order_id=f"ord_{uuid5(NAMESPACE_URL, f'{user_id}:{offset}').hex[:16]}",
        )
        target_status = _STATUSES[offset % len(_STATUSES)]
        if target_status != "pending" and order["version"] == 1:
            repository.update_order(
                user_id=user_id,
                order_id=order["id"],
                expected_version=1,
                status=target_status,
            )
    return days


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the isolated order demo database.")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--days", type=int, default=30, choices=range(1, 91))
    args = parser.parse_args()

    engine = create_order_engine(Settings.from_environment().order_database_url)
    init_order_db(engine)
    created = seed_demo_orders(
        OrderRepository(engine),
        user_id=args.user_id,
        days=args.days,
    )
    print(f"Seeded {created} demo-order days for user {args.user_id}.")


if __name__ == "__main__":
    main()
