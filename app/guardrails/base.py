"""Shared guardrail types and a fail-fast runner.

Rails stay independent classes. This module only defines the result shape
and how a sequence of rails is executed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GuardrailResult:
    """Outcome of one rail. `action` is allow, block, or anonymize."""

    passed: bool
    reason: str | None = None
    action: str = "allow"
    metadata: dict[str, Any] = field(default_factory=dict)


def allow(*, metadata: dict[str, Any] | None = None) -> GuardrailResult:
    return GuardrailResult(passed=True, action="allow", metadata=metadata or {})


def reject(
    reason: str,
    *,
    action: str = "block",
    metadata: dict[str, Any] | None = None,
) -> GuardrailResult:
    return GuardrailResult(
        passed=False,
        reason=reason,
        action=action,
        metadata=metadata or {},
    )


class BaseInputGuardrail(ABC):
    @abstractmethod
    async def check(self, query: str) -> GuardrailResult:
        raise NotImplementedError


class BaseOutputGuardrail(ABC):
    @abstractmethod
    async def check(
        self,
        query: str,
        response: str,
        context: str | None = None,
    ) -> GuardrailResult:
        raise NotImplementedError


async def run_input_rails(
    rails: Sequence[tuple[str, BaseInputGuardrail | None]],
    query: str,
) -> GuardrailResult:
    """Run input rails in order. First failure wins; tag `metadata['rail']`."""
    for name, rail in rails:
        if rail is None:
            continue
        result = await rail.check(query)
        if not result.passed:
            result.metadata.setdefault("rail", name)
            return result
    return allow()


async def run_output_rails(
    rails: Sequence[tuple[str, BaseOutputGuardrail | None]],
    query: str,
    response: str,
    context: str | None = None,
) -> GuardrailResult:
    """Run output rails in order. First failure wins. Does not add extra tags."""
    for _name, rail in rails:
        if rail is None:
            continue
        result = await rail.check(query=query, response=response, context=context)
        if not result.passed:
            return result
    return allow()
