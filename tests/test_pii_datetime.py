"""Dates and times must not block a chat turn as PII."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.agent.multi_agent.nodes import InputGuardrailNode
from app.guardrails.base import GuardrailResult
from app.guardrails.provider.presidio import IGNORED_PII_ENTITIES, PresidioProvider


class _Analyzer:
    def analyze(self, text: str, entities: list[str] | None, language: str):
        return [
            SimpleNamespace(entity_type="DATE_TIME"),
            SimpleNamespace(entity_type="EMAIL_ADDRESS"),
        ]


class _Anonymizer:
    def anonymize(self, text: str, analyzer_results: list) -> SimpleNamespace:
        kept = [item.entity_type for item in analyzer_results]
        return SimpleNamespace(text=f"{text}|{','.join(kept)}")


class _AnonymizingRails:
    async def validate(self, query: str) -> GuardrailResult:
        return GuardrailResult(
            passed=False,
            reason="PII detected: EMAIL_ADDRESS",
            action="anonymize",
            metadata={"anonymized_text": "email me at <EMAIL_ADDRESS>"},
        )


class _BlockingRails:
    async def validate(self, query: str) -> GuardrailResult:
        return GuardrailResult(passed=False, reason="jailbreak", action="block")


def test_date_time_is_not_treated_as_pii() -> None:
    provider = PresidioProvider(analyzer=_Analyzer(), anonymizer=_Anonymizer())
    result = provider.analyze_and_anonymize("meet tomorrow at noon, email a@b.com")
    assert "DATE_TIME" in IGNORED_PII_ENTITIES
    assert result["entities"] == ["EMAIL_ADDRESS"]
    assert "DATE_TIME" not in result["anonymized_text"]


def test_anonymized_input_continues() -> None:
    update = asyncio.run(InputGuardrailNode(_AnonymizingRails())({"query": "raw"}))
    assert update["input_guardrail_passed"] is True
    assert update["query"] == "email me at <EMAIL_ADDRESS>"
    assert update["input_guardrail_reason"] is None


def test_blocked_input_still_stops() -> None:
    update = asyncio.run(InputGuardrailNode(_BlockingRails())({"query": "raw"}))
    assert update["input_guardrail_passed"] is False
    assert update["input_guardrail_reason"] == "jailbreak"
