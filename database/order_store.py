"""Tenant-scoped repository for the isolated order database."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import Engine, event, func, select, update
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload, sessionmaker
from sqlalchemy.pool import NullPool, StaticPool

from database.order_models import Order, OrderBase, OrderItem

ORDER_STATUSES = frozenset({"pending", "confirmed", "processing", "shipped", "cancelled"})


def create_order_engine(database_url: str) -> Engine:
    """Create an order engine with safe SQLite concurrency defaults."""
    from sqlalchemy import create_engine

    url = make_url(database_url)
    if url.get_backend_name() != "sqlite":
        raise ValueError("ORDER_DATABASE_URL must be a SQLite URL")
    if url.database and url.database != ":memory:":
        Path(url.database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False, "timeout": 10},
        poolclass=StaticPool if url.database == ":memory:" else NullPool,
        pool_pre_ping=True,
    )

    @event.listens_for(engine, "connect")
    def _configure_sqlite(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=10000")
        if url.database != ":memory:":
            cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

    return engine


def init_order_db(engine: Engine) -> None:
    """Create the isolated order schema."""
    OrderBase.metadata.create_all(engine)


class OrderRepository:
    """Parameterised order reads/writes with mandatory tenant ownership."""

    def __init__(self, engine: Engine) -> None:
        self._session_factory = sessionmaker(
            bind=engine,
            autoflush=False,
            expire_on_commit=False,
        )

    @staticmethod
    def _same_create_request(
        order: Order,
        *,
        items: Sequence[dict[str, Any]],
        shipping_address: str,
        currency: str,
        note: str | None,
    ) -> bool:
        expected_items = [
            (
                str(item["sku"]),
                str(item["name"]),
                int(item["quantity"]),
                int(item["unit_price_cents"]),
            )
            for item in items
        ]
        actual_items = [
            (item.sku, item.name, item.quantity, item.unit_price_cents)
            for item in order.items
        ]
        return (
            actual_items == expected_items
            and order.shipping_address == shipping_address
            and order.currency == currency.upper()
            and order.note == note
        )

    @staticmethod
    def _serialize(order: Order) -> dict[str, Any]:
        return {
            "id": order.id,
            "user_id": order.user_id,
            "status": order.status,
            "currency": order.currency,
            "total_cents": order.total_cents,
            "shipping_address": order.shipping_address,
            "note": order.note,
            "version": order.version,
            "created_at": order.created_at.isoformat() if order.created_at else None,
            "updated_at": order.updated_at.isoformat() if order.updated_at else None,
            "items": [
                {
                    "sku": item.sku,
                    "name": item.name,
                    "quantity": item.quantity,
                    "unit_price_cents": item.unit_price_cents,
                }
                for item in order.items
            ],
        }

    def get_order(self, *, user_id: str, order_id: str) -> dict[str, Any] | None:
        with self._session_factory() as session:
            order = session.scalar(
                select(Order)
                .options(selectinload(Order.items))
                .where(Order.id == order_id, Order.user_id == user_id)
            )
            return self._serialize(order) if order else None

    def list_orders(
        self,
        *,
        user_id: str,
        days: int = 10,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
        statement = (
            select(Order)
            .options(selectinload(Order.items))
            .where(Order.user_id == user_id, Order.created_at >= cutoff)
            .order_by(Order.created_at.desc())
            .limit(limit)
        )
        if status:
            statement = statement.where(Order.status == status)
        with self._session_factory() as session:
            return [self._serialize(order) for order in session.scalars(statement)]

    def analytics(
        self,
        *,
        user_id: str,
        days: int = 10,
    ) -> dict[str, Any]:
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
        with self._session_factory() as session:
            rows = session.execute(
                select(
                    Order.status,
                    func.count(Order.id),
                    func.coalesce(func.sum(Order.total_cents), 0),
                )
                .where(Order.user_id == user_id, Order.created_at >= cutoff)
                .group_by(Order.status)
            ).all()
        by_status = {
            status: {"orders": int(count), "total_cents": int(total)}
            for status, count, total in rows
        }
        return {
            "days": days,
            "orders": sum(item["orders"] for item in by_status.values()),
            "total_cents": sum(item["total_cents"] for item in by_status.values()),
            "by_status": by_status,
        }

    def create_order(
        self,
        *,
        user_id: str,
        items: Sequence[dict[str, Any]],
        shipping_address: str,
        currency: str,
        note: str | None,
        idempotency_key: str,
        created_at: datetime | None = None,
        order_id: str | None = None,
    ) -> dict[str, Any]:
        if not items:
            raise ValueError("An order requires at least one item")
        if any(
            int(item["quantity"]) < 1 or int(item["unit_price_cents"]) < 0
            for item in items
        ):
            raise ValueError("Order item quantity and price are invalid")
        try:
            with self._session_factory.begin() as session:
                existing = session.scalar(
                    select(Order)
                    .options(selectinload(Order.items))
                    .where(
                        Order.idempotency_key == idempotency_key,
                        Order.user_id == user_id,
                    )
                )
                if existing:
                    if not self._same_create_request(
                        existing,
                        items=items,
                        shipping_address=shipping_address,
                        currency=currency,
                        note=note,
                    ):
                        raise ValueError(
                            "Idempotency key was already used for a different order"
                        )
                    return self._serialize(existing)
                order = Order(
                    id=order_id or f"ord_{uuid4().hex[:16]}",
                    user_id=user_id,
                    status="pending",
                    currency=currency.upper(),
                    total_cents=sum(
                        int(item["quantity"]) * int(item["unit_price_cents"])
                        for item in items
                    ),
                    shipping_address=shipping_address,
                    note=note,
                    idempotency_key=idempotency_key,
                    created_at=created_at or datetime.now(timezone.utc).replace(tzinfo=None),
                )
                order.items = [
                    OrderItem(
                        sku=str(item["sku"]),
                        name=str(item["name"]),
                        quantity=int(item["quantity"]),
                        unit_price_cents=int(item["unit_price_cents"]),
                    )
                    for item in items
                ]
                session.add(order)
                session.flush()
                return self._serialize(order)
        except IntegrityError:
            with self._session_factory() as session:
                existing = session.scalar(
                    select(Order)
                    .options(selectinload(Order.items))
                    .where(
                        Order.idempotency_key == idempotency_key,
                        Order.user_id == user_id,
                    )
                )
                if existing is None:
                    raise
                if not self._same_create_request(
                    existing,
                    items=items,
                    shipping_address=shipping_address,
                    currency=currency,
                    note=note,
                ):
                    raise ValueError(
                        "Idempotency key was already used for a different order"
                    )
                return self._serialize(existing)

    def update_order(
        self,
        *,
        user_id: str,
        order_id: str,
        expected_version: int,
        status: str | None = None,
        shipping_address: str | None = None,
        note: str | None = None,
    ) -> dict[str, Any] | None:
        if status and status not in ORDER_STATUSES:
            raise ValueError(f"Unsupported order status: {status}")
        with self._session_factory.begin() as session:
            changes: dict[str, Any] = {
                "version": Order.version + 1,
                "updated_at": datetime.now(timezone.utc).replace(tzinfo=None),
            }
            if status is not None:
                changes["status"] = status
            if shipping_address is not None:
                changes["shipping_address"] = shipping_address
            if note is not None:
                changes["note"] = note
            result = session.execute(
                update(Order)
                .where(
                    Order.id == order_id,
                    Order.user_id == user_id,
                    Order.version == expected_version,
                )
                .values(**changes)
            )
            if result.rowcount != 1:
                return None
            order = session.scalar(
                select(Order)
                .options(selectinload(Order.items))
                .where(Order.id == order_id, Order.user_id == user_id)
            )
            if order is None:
                return None
            return self._serialize(order)
