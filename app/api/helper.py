
from app.models.schemas import (
    DocumentResponse,
    normalize_metadata,
    normalize_tags,
)
import shutil
import hashlib
from app.core.config import Settings
settings = Settings.from_environment()
from pathlib import Path
from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session
from sqlalchemy import select
from database.models import Document
import logging
import uuid
import json
from typing import Any
import asyncio
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic




logger = logging.getLogger(__name__)

_DOCUMENTS_ROOT = Path("documents")

def _content_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _parse_metadata_form(raw: str | None) -> dict[str, str]:
    """Accept a JSON object or comma-separated key:value pairs."""
    if not raw or not raw.strip():
        return {}
    try:
        return normalize_metadata(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _parse_tags_form(raw: str | None) -> list[str]:
    """Accept a JSON array or a comma-separated string from the upload form."""
    if not raw or not raw.strip():
        return []
    text = raw.strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="tags must be a JSON array") from exc
        if not isinstance(parsed, list):
            raise HTTPException(status_code=400, detail="tags must be a JSON array")
        return normalize_tags(parsed)
    return normalize_tags(text.split(","))


def _row_tags(row: Document) -> list[str]:
    """Decode the JSON tags column; tolerate legacy rows without tags."""
    if not row.tags:
        return []
    try:
        parsed = json.loads(row.tags)
    except json.JSONDecodeError:
        return []
    return normalize_tags(parsed) if isinstance(parsed, list) else []


def _row_metadata(row: Document) -> dict[str, str]:
    if not row.custom_metadata:
        return {}
    try:
        parsed = json.loads(row.custom_metadata)
    except json.JSONDecodeError:
        return {}
    return normalize_metadata(parsed) if isinstance(parsed, dict) else {}


def _document_response(row: Document, *, chunk_count: int, file_size: int | None) -> DocumentResponse:
    status = row.status if row.status in {"processing", "ready", "failed"} else "ready"
    return DocumentResponse(
        id=row.id,
        user_id=row.user_id,
        file_name=row.file_name,
        created_at=row.created_at,
        tags=_row_tags(row),
        metadata=_row_metadata(row),
        file_size=file_size,
        chunk_count=chunk_count,
        status=status,
        error_message=row.error_message,
    )


def _find_duplicate(db: Session, *, user_id: str, content_hash: str) -> Document | None:
    return db.scalar(select(Document).where(Document.user_id == user_id,Document.content_hash == content_hash))


def _mark_failed(db: Session, row: Document, file_path: str | None) -> None:
    row.status = "failed"
    row.error_message = "Ingestion failed."
    row.content_hash = None
    _delete_stored_file(file_path)
    db.commit()


def _stored_file_size(file_path: str | None) -> int | None:
    stored = _resolve_stored_file(file_path)
    if stored is None or not stored.is_file():
        return None
    try:
        return stored.stat().st_size
    except OSError:
        return None


def _save_upload(file: UploadFile,content: bytes,*,user_id: str,document_id: str) -> str:
    """Save the upload under documents/{user_id}/{document_id}{ext}."""
    suffix = Path(file.filename or "upload").suffix
    dest_dir = _DOCUMENTS_ROOT / user_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{document_id}{suffix}"
    dest.write_bytes(content)
    return str(dest)


def _resolve_stored_file(file_path: str | None) -> Path | None:
    """Return a path only if it exists under the documents directory."""
    if not file_path:
        return None

    try:
        resolved = Path(file_path).resolve()
        root = _DOCUMENTS_ROOT.resolve()
        if root not in resolved.parents:
            return None
        return resolved
    except OSError:
        return None


def _delete_stored_file(file_path: str | None) -> None:
    """Remove a stored upload only if it lives under the documents directory."""
    resolved = _resolve_stored_file(file_path)
    if resolved is None:
        return

    try:
        resolved.unlink(missing_ok=True)
        images = resolved.parent / f"{resolved.stem}_images"
        if images.is_dir() and _DOCUMENTS_ROOT.resolve() in images.resolve().parents:
            shutil.rmtree(images, ignore_errors=True)
    except OSError:
        logger.warning("Could not remove stored upload %s", resolved, exc_info=True)

def generate_new_session_id() -> str:
    return f"session_{uuid.uuid4()}"


def _chat_run_name(question: str | None) -> str:
    """LangSmith lists a run by name, so the name is the user question."""
    label = " ".join((question or "").split())
    if not label:
        return "chat"
    if len(label) > 120:
        return label[:117] + "..."
    return label



def _build_graph_config(thread_id: str,user_id: str,question: str | None = None) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": thread_id},
        "run_name": _chat_run_name(question),
        "tags": ["chat"],
        "metadata": {
            "source": "fastapi",
            "user_id": user_id,
            "session_id": thread_id,
            "thread_id": thread_id,
        },
    }


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _display_tokens(text: str) -> list[str]:
    """Break a finished string into word-sized SSE chunks."""
    if not text:
        return []
    chunks: list[str] = []
    current: list[str] = []
    for char in text:
        current.append(char)
        if char.isspace():
            chunks.append("".join(current))
            current = []
    if current:
        chunks.append("".join(current))
    return chunks


async def _iter_display_sse(text: str):
    """Yield token SSE frames slowly enough for the UI to paint each word."""
    for part in _display_tokens(text):
        yield _sse({"type": "token", "content": part})
        await asyncio.sleep(0.02)






async def _run_agent(master_graph: Any,question: str,user_id: str,thread_id: str) -> dict[str, Any]:
    async with asyncio.timeout(60):
        result = await master_graph.ainvoke(
            {
                "query": question,
                "user_id": user_id,
                "thread_id": thread_id,
                "messages": [HumanMessage(content=question)],
                "iterations": 0,
                "max_iterations": 2,
            },
            config=_build_graph_config(thread_id, user_id, question),
        )
    trajectory = [
        {
            "agent_name": out.agent_name,
            "result": out.result,
            "metadata": out.metadata,
        }
        for out in result.get("agent_outputs", [])
    ]
    direct_response = any(
        bool((item["metadata"] or {}).get("direct_response"))
        for item in trajectory
    )
    return {
        "answer": result.get("final_response") or "Unable to synthesize answer.",
        "iterations": result.get("iterations", 1),
        "agent_trajectory": trajectory,
        "guardrail_metadata": result.get("guardrail_metadata", {}),
        "long_term_memories": result.get("long_term_memories") or [],
        "direct_response": direct_response,
        "input_guardrail_passed": result.get("input_guardrail_passed", True),
        "sources": _sources_from_trajectory(trajectory),
    }


def _sources_from_trajectory(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Citation targets ({index, filename, page, chunk_id}) emitted by the RAG specialist."""
    sources: list[dict[str, Any]] = []
    for item in trajectory:
        meta = item.get("metadata") or {}
        for source in meta.get("sources") or []:
            if isinstance(source, dict):
                sources.append(source)
    return sources


def _context_from_agent(agent_result: dict[str, Any]) -> str:
    """Build Claude context from specialist observations (RAG chunks, web)."""
    parts: list[str] = []
    for item in agent_result.get("agent_trajectory") or []:
        meta = item.get("metadata") or {}
        blob = (meta.get("context") or item.get("result") or "").strip()
        if not blob:
            continue
        name = item.get("agent_name") or "agent"
        parts.append(f"[{name}]\n{blob}")
    return "\n\n".join(parts)

