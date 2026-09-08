import json
from app.guardrails.base import BaseOutputGuardrail, GuardrailResult

class OutputSchemaGuardrail(BaseOutputGuardrail):
    def __init__(self, schema=None):
        self.schema = schema

    async def check(self, query: str, response: str, context: str | None = None) -> GuardrailResult:
        if not self.schema:
            return GuardrailResult(passed=True, action="allow")

        try:
            parsed_json = json.loads(response)
            # Add pydantic or jsonschema validation logic here if provided
            return GuardrailResult(passed=True, action="allow", metadata={"parsed": parsed_json})
        except json.JSONDecodeError as e:
            return GuardrailResult(
                passed=False,
                reason=f"Output is not valid JSON against expected schema: {str(e)}",
                action="block"
            )