from app.guardrails.base import GuardrailResult, BaseOutputGuardrail

class OutputGuardrailService:
    def __init__(
        self,
        schema_validator: BaseOutputGuardrail | None = None,
        grounding_checker: BaseOutputGuardrail | None = None,
        pii_detector: BaseOutputGuardrail | None = None,
        safety_checker: BaseOutputGuardrail | None = None,
    ):
        self.schema_validator = schema_validator
        self.grounding_checker = grounding_checker
        self.pii_detector = pii_detector
        self.safety_checker = safety_checker

    async def validate(self, query: str, response: str, context: str | None = None) -> GuardrailResult:
        checks = [
            self.schema_validator,
            self.grounding_checker,
            self.pii_detector,
            self.safety_checker,
        ]

        for guardrail in checks:
            if guardrail is None:
                continue

            result = await guardrail.check(
                query=query,
                response=response,
                context=context,
            )

            # If any output check fails, stop and return the failure immediately
            if not result.passed:
                return result

        return GuardrailResult(
            passed=True,
            action="allow",
        )