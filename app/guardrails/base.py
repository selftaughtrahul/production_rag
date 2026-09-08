from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GuardrailResult:
    ''' '''
    passed: bool
    reason: str | None = None
    action: str = "allow"
    metadata: dict[str, Any] = field(default_factory=dict)


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