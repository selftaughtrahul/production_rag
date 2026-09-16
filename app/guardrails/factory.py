"""Construct the wired input/output rails. Swap providers here, not in nodes."""

from __future__ import annotations

from typing import Any

from app.guardrails.fast_checks import provider_is_ready
from app.guardrails.input.injection import PromptInjectionGuardrail
from app.guardrails.input.jailbreak import JailbreakGuardrail
from app.guardrails.input.pii import PresidioPIIGuardrail as InputPresidioGuardrail
from app.guardrails.input.safety import InputSafetyGuardrail
from app.guardrails.input.validation import InputValidationGuardrail
from app.guardrails.input_service import InputGuardrailService
from app.guardrails.output.grounding import GroundingGuardrail
from app.guardrails.output.pii import OutputPIIGuardrail
from app.guardrails.output.safety import OutputSafetyGuardrail
from app.guardrails.output.schema import OutputSchemaGuardrail
from app.guardrails.output_service import OutputGuardrailService


def build_input_guardrails(
    nemo_provider: Any | None = None,
    presidio_provider: Any | None = None,
    llama_guard_provider: Any | None = None,
) -> InputGuardrailService:
    prompt_injection = (
        PromptInjectionGuardrail(provider=nemo_provider)
        if provider_is_ready(nemo_provider)
        else None
    )
    pii_detector = (
        InputPresidioGuardrail(provider=presidio_provider) if presidio_provider else None
    )
    safety_checker = (
        InputSafetyGuardrail(provider=llama_guard_provider)
        if llama_guard_provider
        else None
    )
    return InputGuardrailService(
        input_validator=InputValidationGuardrail(max_length=10_000),
        jailbreak=JailbreakGuardrail(provider=nemo_provider),
        prompt_injection=prompt_injection,
        pii_detector=pii_detector,
        safety_checker=safety_checker,
    )


def build_output_guardrails(
    presidio_provider: Any | None = None,
    llama_guard_provider: Any | None = None,
    grounding_evaluator: Any | None = None,
    response_schema: Any | None = None,
) -> OutputGuardrailService:
    schema_validator = (
        OutputSchemaGuardrail(schema=response_schema) if response_schema else None
    )
    grounding_checker = (
        GroundingGuardrail(evaluator=grounding_evaluator) if grounding_evaluator else None
    )
    pii_detector = (
        OutputPIIGuardrail(provider=presidio_provider) if presidio_provider else None
    )
    safety_checker = (
        OutputSafetyGuardrail(provider=llama_guard_provider)
        if llama_guard_provider
        else None
    )
    return OutputGuardrailService(
        schema_validator=schema_validator,
        grounding_checker=grounding_checker,
        pii_detector=pii_detector,
        safety_checker=safety_checker,
    )
