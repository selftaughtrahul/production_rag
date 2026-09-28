"""Checkpoint metadata owner parsing."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

import ormsgpack

from app.api.session_access import (
    _decode_msgpack_str,
    _is_user_facing_thread,
    _owner_from_metadata,
)


def test_owner_from_dict() -> None:
    assert _owner_from_metadata({"user_id": "u1"}) == "u1"


def test_owner_from_json_bytes() -> None:
    assert _owner_from_metadata(b'{"user_id": "u2"}') == "u2"


def test_owner_from_memoryview() -> None:
    raw = memoryview(b'{"user_id": "u3"}')
    assert _owner_from_metadata(raw) == "u3"


def test_owner_from_json_str() -> None:
    assert _owner_from_metadata('{"user_id": "u4"}') == "u4"


def test_owner_ignores_non_dict_json() -> None:
    assert _owner_from_metadata("[1, 2]") is None
    assert _owner_from_metadata(b"not-json") is None
    assert _owner_from_metadata(None) is None


def test_decode_msgpack_str_reads_user_query() -> None:
    packed = ormsgpack.packb("show my orders")
    assert _decode_msgpack_str(packed) == "show my orders"
    assert _decode_msgpack_str(memoryview(packed)) == "show my orders"
    assert _decode_msgpack_str(b"") is None
    assert _decode_msgpack_str(ormsgpack.packb(12)) is None


def test_nested_rag_threads_are_not_user_facing() -> None:
    assert _is_user_facing_thread("session_abc") is True
    assert _is_user_facing_thread("session_abc:rag") is False
