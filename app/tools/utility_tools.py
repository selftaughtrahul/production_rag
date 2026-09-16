"""Deterministic general-agent tools for arithmetic and current time."""

from __future__ import annotations

import ast
import operator
from datetime import datetime
from typing import Callable, Type
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field

from app.tools.base import BaseAgentTool

_BINARY_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 12:
            raise ValueError("Exponent is too large")
        return _BINARY_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate(node.operand))
    raise ValueError("Expression contains unsupported syntax")


class CalculatorInput(BaseModel):
    expression: str = Field(min_length=1, max_length=200)


class CalculatorTool(BaseAgentTool):
    name: str = "calculate"
    description: str = "Evaluate a basic arithmetic expression without executing code."
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(self, expression: str) -> str:
        try:
            value = _evaluate(ast.parse(expression, mode="eval"))
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as exc:
            return self._format_error(str(exc)).to_str()
        result: int | float = int(value) if value.is_integer() else value
        return self._format_success({"expression": expression, "result": result}).to_str()

    async def _arun(self, expression: str) -> str:
        return self._run(expression)


class CurrentDateTimeInput(BaseModel):
    timezone_name: str = Field(default="UTC", min_length=1, max_length=100)


class CurrentDateTimeTool(BaseAgentTool):
    name: str = "current_datetime"
    description: str = "Return the current date and time in an IANA timezone."
    args_schema: Type[BaseModel] = CurrentDateTimeInput

    def _run(self, timezone_name: str = "UTC") -> str:
        try:
            zone = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError:
            return self._format_error("Unknown IANA timezone.").to_str()
        now = datetime.now(zone)
        return self._format_success(
            {
                "timezone": timezone_name,
                "iso": now.isoformat(),
                "date": now.date().isoformat(),
                "time": now.strftime("%H:%M:%S"),
            }
        ).to_str()

    async def _arun(self, timezone_name: str = "UTC") -> str:
        return self._run(timezone_name)
