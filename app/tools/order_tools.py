"""Typed, tenant-scoped tools for order reads, analytics, and writes."""

from __future__ import annotations

import asyncio
from typing import Type

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.json_schema import SkipJsonSchema

from app.tools.base import BaseAgentTool
from database.order_store import ORDER_STATUSES, OrderRepository


class OrderItemInput(BaseModel):
    sku: str = Field(min_length=1, max_length=80)
    name: str = Field(min_length=1, max_length=200)
    quantity: int = Field(ge=1, le=100)
    unit_price_cents: int = Field(ge=0, le=100_000_000)


class GetOrderInput(BaseModel):
    order_id: str = Field(min_length=5, max_length=100)
    user_id: SkipJsonSchema[str] = Field(min_length=1)


class ListOrdersInput(BaseModel):
    user_id: SkipJsonSchema[str] = Field(min_length=1)
    days: int = Field(default=10, ge=1, le=365)
    status: str | None = None
    limit: int = Field(default=50, ge=1, le=100)

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str | None) -> str | None:
        if value is not None and value not in ORDER_STATUSES:
            raise ValueError("Unsupported order status")
        return value


class OrderAnalyticsInput(BaseModel):
    user_id: SkipJsonSchema[str] = Field(min_length=1)
    days: int = Field(default=10, ge=1, le=365)


class CreateOrderInput(BaseModel):
    user_id: SkipJsonSchema[str] = Field(min_length=1)
    items: list[OrderItemInput] = Field(min_length=1, max_length=50)
    shipping_address: str = Field(min_length=5, max_length=500)
    currency: str = Field(default="USD", min_length=3, max_length=3)
    note: str | None = Field(default=None, max_length=1000)
    idempotency_key: str = Field(min_length=8, max_length=100)


class UpdateOrderInput(BaseModel):
    user_id: SkipJsonSchema[str] = Field(min_length=1)
    order_id: str = Field(min_length=5, max_length=100)
    expected_version: int = Field(ge=1)
    status: str | None = None
    shipping_address: str | None = Field(default=None, min_length=5, max_length=500)
    note: str | None = Field(default=None, max_length=1000)

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str | None) -> str | None:
        if value is not None and value not in ORDER_STATUSES:
            raise ValueError("Unsupported order status")
        return value


class _OrderTool(BaseAgentTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    repository: OrderRepository


class GetOrderTool(_OrderTool):
    name: str = "get_order"
    description: str = "Get one authenticated user's order by order ID."
    args_schema: Type[BaseModel] = GetOrderInput

    def _run(self, order_id: str, user_id: str) -> str:
        order = self.repository.get_order(user_id=user_id, order_id=order_id)
        if order is None:
            return self._format_error("Order not found.").to_str()
        return self._format_success(order).to_str()

    async def _arun(self, order_id: str, user_id: str) -> str:
        return await asyncio.to_thread(self._run, order_id, user_id)


class ListOrdersTool(_OrderTool):
    name: str = "list_orders"
    description: str = "List the authenticated user's recent orders."
    args_schema: Type[BaseModel] = ListOrdersInput

    def _run(
        self,
        user_id: str,
        days: int = 10,
        status: str | None = None,
        limit: int = 50,
    ) -> str:
        orders = self.repository.list_orders(
            user_id=user_id,
            days=days,
            status=status,
            limit=limit,
        )
        return self._format_success(orders, {"rows_returned": len(orders)}).to_str()

    async def _arun(
        self,
        user_id: str,
        days: int = 10,
        status: str | None = None,
        limit: int = 50,
    ) -> str:
        return await asyncio.to_thread(self._run, user_id, days, status, limit)


class OrderAnalyticsTool(_OrderTool):
    name: str = "order_analytics"
    description: str = "Aggregate the authenticated user's order count and value by status."
    args_schema: Type[BaseModel] = OrderAnalyticsInput

    def _run(self, user_id: str, days: int = 10) -> str:
        return self._format_success(
            self.repository.analytics(user_id=user_id, days=days)
        ).to_str()

    async def _arun(self, user_id: str, days: int = 10) -> str:
        return await asyncio.to_thread(self._run, user_id, days)


class CreateOrderTool(_OrderTool):
    name: str = "create_order"
    description: str = "Create an order after explicit human confirmation."
    args_schema: Type[BaseModel] = CreateOrderInput
    requires_approval: bool = True

    def _run(
        self,
        user_id: str,
        items: list[OrderItemInput],
        shipping_address: str,
        currency: str = "USD",
        note: str | None = None,
        idempotency_key: str = "",
    ) -> str:
        order = self.repository.create_order(
            user_id=user_id,
            items=[
                item.model_dump() if isinstance(item, OrderItemInput) else item
                for item in items
            ],
            shipping_address=shipping_address,
            currency=currency,
            note=note,
            idempotency_key=idempotency_key,
        )
        return self._format_success(order).to_str()

    async def _arun(self, **kwargs) -> str:
        return await asyncio.to_thread(self._run, **kwargs)


class UpdateOrderTool(_OrderTool):
    name: str = "update_order"
    description: str = "Update an owned order after explicit human confirmation."
    args_schema: Type[BaseModel] = UpdateOrderInput
    requires_approval: bool = True

    def _run(
        self,
        user_id: str,
        order_id: str,
        expected_version: int,
        status: str | None = None,
        shipping_address: str | None = None,
        note: str | None = None,
    ) -> str:
        order = self.repository.update_order(
            user_id=user_id,
            order_id=order_id,
            expected_version=expected_version,
            status=status,
            shipping_address=shipping_address,
            note=note,
        )
        if order is None:
            return self._format_error(
                "Order not found or its version changed; fetch it again."
            ).to_str()
        return self._format_success(order).to_str()

    async def _arun(self, **kwargs) -> str:
        return await asyncio.to_thread(self._run, **kwargs)
