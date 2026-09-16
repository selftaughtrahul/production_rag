"""SQL tool: read-only checks and denied-table allowlist."""

from __future__ import annotations

from sqlalchemy import create_engine

from app.tools.sql_search import SQLQueryTool, _referenced_tables


def _tool() -> SQLQueryTool:
    return SQLQueryTool(db_engine=create_engine("sqlite:///:memory:"))


def test_select_is_read_only() -> None:
    tool = _tool()
    assert tool._is_read_only("SELECT id FROM documents") is True
    assert tool._is_read_only("WITH x AS (SELECT 1) SELECT * FROM x") is True


def test_writes_and_stacked_statements_are_blocked() -> None:
    tool = _tool()
    assert tool._is_read_only("DELETE FROM documents") is False
    assert tool._is_read_only("DROP TABLE documents") is False
    assert tool._is_read_only("SELECT 1; DROP TABLE users") is False
    assert tool._is_read_only("SELECT 1; SELECT 2") is False


def test_comment_cannot_hide_write() -> None:
    tool = _tool()
    assert tool._is_read_only("SELECT 1; --\nDELETE FROM users") is False


def test_denied_tables_are_detected() -> None:
    assert "users" in _referenced_tables("SELECT * FROM users")
    assert "user_memories" in _referenced_tables(
        'SELECT memory FROM "user_memories"'
    )


def test_query_on_users_is_rejected() -> None:
    result = _tool()._run("SELECT * FROM users")
    assert "restricted table" in result.lower()


def test_non_select_error_message() -> None:
    result = _tool()._run("UPDATE documents SET file_name = 'x'")
    assert "read-only" in result.lower()
