"""
"""

from app.guardrails.model import GuardrailResult

class InputGuardrailService:

    ''' '''


    def __init__(self, prompt_injection=None, pii_detector=None, safety_checker=None, input_validator=None,):
        self.prompt_injection = prompt_injection
        self.pii_detector = pii_detector
        self.safety_checker = safety_checker
        self.input_validator = input_validator

    async def validate(self, query: str) -> GuardrailResult:

        checks = [
            self.input_validator,
            self.prompt_injection,
            self.pii_detector,
            self.safety_checker,
        ]

        for guardrail in checks:

            if guardrail is None:
                continue

            result = await guardrail.check(query)

            if not result.passed:
                return result

        return GuardrailResult(
            passed=True,
            action="allow",
        )