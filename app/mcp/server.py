"""MCP stdio server: health + ask_rag (HTTP to this API).

Run: python -m app.mcp.server

Cursor MCP config: command python, args ["-m", "app.mcp.server"]
Env: API_BASE_URL, MCP_JWT (Bearer for POST /chat/).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

_PROTOCOL = "2024-11-05"


def _api_base() -> str:
    return os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def _auth_headers() -> dict[str, str]:
    token = os.getenv("MCP_JWT", "").strip()
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def health() -> str:
    """Liveness of the MCP wrapper (does not load RAG models)."""
    return "ok"


def ask_rag(question: str, session_id: str | None = None) -> str:
    """Ask the production RAG chat API and return the assembled SSE answer."""
    if not question.strip():
        return "Question cannot be empty."
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        **_auth_headers(),
    }
    if "Authorization" not in headers:
        return "Set MCP_JWT to a valid API bearer token."
    body: dict[str, str] = {"question": question.strip()}
    if session_id:
        body["session_id"] = session_id
    pieces: list[str] = []
    try:
        import httpx

        with httpx.Client(timeout=120.0) as client:
            with client.stream(
                "POST",
                f"{_api_base()}/chat/",
                headers=headers,
                json=body,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = json.loads(line[6:])
                    kind = payload.get("type")
                    if kind == "token":
                        pieces.append(str(payload.get("content") or ""))
                    elif kind == "done":
                        return str(payload.get("answer") or "".join(pieces))
                    elif kind == "error":
                        return str(payload.get("error") or "Chat error")
    except Exception as exc:
        return f"MCP ask_rag failed: {exc}"
    return "".join(pieces).strip() or "No answer."


def _tools_list() -> list[dict[str, Any]]:
    return [
        {
            "name": "health",
            "description": "Liveness of the MCP wrapper.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "ask_rag",
            "description": "Ask the RAG chat API and return the answer.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "session_id": {"type": "string"},
                },
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    ]


def _call_tool(name: str, arguments: dict[str, Any]) -> str:
    if name == "health":
        return health()
    if name == "ask_rag":
        return ask_rag(
            str(arguments.get("question") or ""),
            session_id=arguments.get("session_id"),
        )
    return f"Unknown tool: {name}"


def _read_message() -> dict[str, Any] | None:
    headers: dict[str, str] = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        decoded = line.decode("utf-8")
        key, _, value = decoded.partition(":")
        headers[key.strip().lower()] = value.strip()
    length = int(headers.get("content-length", "0"))
    body = sys.stdin.buffer.read(length)
    if not body:
        return None
    return json.loads(body.decode("utf-8"))


def _write_message(payload: dict[str, Any]) -> None:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(raw)}\r\n\r\n".encode("ascii") + raw)
    sys.stdout.buffer.flush()


def _handle(message: dict[str, Any]) -> dict[str, Any] | None:
    method = message.get("method")
    msg_id = message.get("id")
    params = message.get("params") or {}
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": _PROTOCOL,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "basic_rag", "version": "2.0.0"},
            },
        }
    if method == "notifications/initialized":
        return None
    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}
    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"tools": _tools_list()},
        }
    if method == "tools/call":
        name = str(params.get("name") or "")
        arguments = params.get("arguments") or {}
        text = _call_tool(name, arguments if isinstance(arguments, dict) else {})
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "content": [{"type": "text", "text": text}],
                "isError": text.startswith("Unknown tool") or text.startswith("MCP ask_rag failed"),
            },
        }
    if msg_id is None:
        return None
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def main() -> None:
    while True:
        message = _read_message()
        if message is None:
            return
        reply = _handle(message)
        if reply is not None:
            _write_message(reply)


if __name__ == "__main__":
    main()
