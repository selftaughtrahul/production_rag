"""Specialist tool safety and allowlists."""

from __future__ import annotations

import asyncio

import pytest

from app.tools.order_tools import CreateOrderInput
from app.tools.registry import AGENT_TOOL_NAMES, ToolRegistry
from app.tools.utility_tools import CalculatorTool, CurrentDateTimeTool
from app.tools.web_read import _validate_public_url


def test_agent_tool_allowlists_are_separate() -> None:
    all_names = [name for names in AGENT_TOOL_NAMES.values() for name in names]
    assert "create_order" in AGENT_TOOL_NAMES["sql_agent"]
    assert "create_order" not in AGENT_TOOL_NAMES["general_agent"]
    assert len(all_names) == len(set(all_names))


def test_registry_returns_only_assigned_tools() -> None:
    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())
    registry.register_tool(CurrentDateTimeTool())
    assert {tool.name for tool in registry.get_tools_for_agent("general_agent")} == {
        "calculate",
        "current_datetime",
    }
    assert registry.get_tools_for_agent("sql_agent") == []
    assert registry.get_tool_for_agent("sql_agent", "calculate") is None


def test_calculator_rejects_code_and_calculates_arithmetic() -> None:
    tool = CalculatorTool()
    result = asyncio.run(tool.ainvoke({"expression": "(2 + 3) * 4"}))
    assert "'result': 20" in result
    denied = asyncio.run(tool.ainvoke({"expression": "__import__('os')"}))
    assert "unsupported syntax" in denied.lower()


def test_current_datetime_validates_timezone() -> None:
    tool = CurrentDateTimeTool()
    result = asyncio.run(tool.ainvoke({"timezone_name": "Asia/Kolkata"}))
    assert "Asia/Kolkata" in result
    invalid = asyncio.run(tool.ainvoke({"timezone_name": "Mars/Olympus"}))
    assert "unknown iana timezone" in invalid.lower()


def test_web_reader_blocks_local_networks() -> None:
    with pytest.raises(ValueError, match="Private"):
        _validate_public_url("http://127.0.0.1/admin")
    with pytest.raises(ValueError, match="HTTP"):
        _validate_public_url("file:///etc/passwd")
    with pytest.raises(ValueError, match="metadata"):
        _validate_public_url("http://metadata.google.internal/")
    with pytest.raises(ValueError, match="credentials"):
        _validate_public_url("https://user:secret@example.com/")


def test_authenticated_user_id_is_not_model_selectable() -> None:
    assert "user_id" not in CreateOrderInput.model_json_schema()["properties"]
