from app.guardrails.input_service import InputGuardrailService
from app.guardrails.output_service import OutputGuardrailService
from app.guardrails.input.validation import InputValidationGuardrail
from app.guardrails.input.injection import PromptInjectionGuardrail
from app.guardrails.input.pii import PresidioPIIGuardrail as InputPresidioGuardrail
from app.guardrails.input.safety import InputSafetyGuardrail
from app.guardrails.output.schema import OutputSchemaGuardrail
from app.guardrails.output.grounding import GroundingGuardrail
from app.guardrails.output.pii import OutputPIIGuardrail
from app.guardrails.output.safety import OutputSafetyGuardrail


def build_input_guardrails(nemo_provider=None, presidio_provider=None, llama_guard_provider=None) -> InputGuardrailService:

    input_validator = InputValidationGuardrail(max_length=10_000)
    prompt_injection = (
        PromptInjectionGuardrail(provider=nemo_provider)
        if nemo_provider is not None and getattr(nemo_provider, "is_ready", False)
        else None
    )
    pii_detector = InputPresidioGuardrail(provider=presidio_provider) if presidio_provider else None
    safety_checker = InputSafetyGuardrail(provider=llama_guard_provider) if llama_guard_provider else None

    return InputGuardrailService(
        input_validator=input_validator,
        prompt_injection=prompt_injection,
        pii_detector=pii_detector,
        safety_checker=safety_checker,
    )


def build_output_guardrails(presidio_provider=None, llama_guard_provider=None, grounding_evaluator=None, response_schema=None) -> OutputGuardrailService:

    schema_validator = OutputSchemaGuardrail(schema=response_schema) if response_schema else None
    grounding_checker = GroundingGuardrail(evaluator=grounding_evaluator) if grounding_evaluator else None
    pii_detector = OutputPIIGuardrail(provider=presidio_provider) if presidio_provider else None
    safety_checker = OutputSafetyGuardrail(provider=llama_guard_provider) if llama_guard_provider else None

    return OutputGuardrailService(
        schema_validator=schema_validator,
        grounding_checker=grounding_checker,
        pii_detector=pii_detector,
        safety_checker=safety_checker,
    )