"""NeMo Guardrails provider for prompt-injection checks."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from app.guardrails.fast_checks import needs_llm_injection_check

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "nemo_config"


def _import_nemo():
    # Anthropic is not OpenAI-compatible; NeMo 0.22+ needs the LangChain framework.
    os.environ.setdefault("NEMOGUARDRAILS_LLM_FRAMEWORK", "langchain")
    from nemoguardrails import LLMRails, RailsConfig

    return LLMRails, RailsConfig


class NeMoProvider:
    def __init__(self, config_path: str | None = None):
        self.rails = None
        path = Path(config_path or _DEFAULT_CONFIG)
        if not path.is_dir():
            logger.warning("NeMo config directory not found: %s", path)
            return
        if not os.getenv("ANTHROPIC_API_KEY", "").strip():
            logger.warning(
                "ANTHROPIC_API_KEY is not set; NeMo Guardrails cannot load the Anthropic engine."
            )
            return

        try:
            LLMRails, RailsConfig = _import_nemo()
            self.rails = LLMRails(RailsConfig.from_path(str(path)))
            logger.info("NeMo Guardrails loaded from %s", path)
        except ImportError:
            logger.warning(
                "nemoguardrails is not installed. "
                "Install it with: pip install nemoguardrails langchain-anthropic"
            )
        except Exception as exc:
            logger.error("Failed to load NeMo configuration: %s", exc)

    @property
    def is_ready(self) -> bool:
        return self.rails is not None

    async def check_prompt_injection(self, query: str) -> dict[str, Any]:
        if not self.rails:
            return _denylist_check(query)

        try:
            response = await self.rails.generate_async(
                messages=[{"role": "user", "content": query}],
                options={"log": {"activated_rails": True}},
            )
            blocked = _is_blocked(response)
            content = _response_text(response)
            return {"passed": not blocked, "content": content or query}
        except Exception as exc:
            logger.error("NeMo rail check failed: %s", exc)
            return {
                "passed": False,
                "content": query,
                "reason": "NeMo unavailable; failing closed.",
            }


def _denylist_check(query: str) -> dict[str, Any]:
    """Keyword fallback when rails are not loaded. Hits are blocked."""
    if needs_llm_injection_check(query):
        return {
            "passed": False,
            "content": query,
            "reason": "Injection pattern matched; NeMo rails are not loaded.",
        }
    return {
        "passed": True,
        "content": query,
        "fallback": "denylist",
    }


def _response_text(response: Any) -> str:
    if isinstance(response, str):
        return response

    payload = getattr(response, "response", None)
    if isinstance(payload, str):
        return payload
    if isinstance(payload, list) and payload:
        last = payload[-1]
        if isinstance(last, dict):
            return str(last.get("content") or "")

    if isinstance(response, dict):
        return str(response.get("content") or "")
    return str(getattr(response, "content", "") or "")


def _is_blocked(response: Any) -> bool:
    log = getattr(response, "log", None)
    for rail in getattr(log, "activated_rails", None) or []:
        if getattr(rail, "stop", False):
            return True
        decisions = [d.lower() for d in getattr(rail, "decisions", []) or []]
        if any("refuse" in decision or decision == "stop" for decision in decisions):
            return True

    text = _response_text(response).lower()
    return any(
        marker in text
        for marker in (
            "i cannot answer this",
            "i can't respond",
            "i cannot respond",
            "i'm sorry, i can't",
        )
    )
