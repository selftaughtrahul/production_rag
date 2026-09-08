from app.guardrails.base import BaseOutputGuardrail, GuardrailResult


class GroundingGuardrail(BaseOutputGuardrail):
    def __init__(self, evaluator=None, threshold: float = 0.7):
        self.evaluator = evaluator
        self.threshold = threshold

    async def check(self, query: str, response: str, context: str | None = None) -> GuardrailResult:
        if not context:
            # If no retrieved RAG context exists, skip grounding evaluation
            return GuardrailResult(passed=True, action="allow")

        if self.evaluator:
            score = await self.evaluator.evaluate(response=response, context=context)
            if score < self.threshold:
                return GuardrailResult(
                    passed=False,
                    reason=f"Grounding score {score:.2f} below threshold {self.threshold}.",
                    action="block",
                    metadata={"grounding_score": score}
                )

        return GuardrailResult(passed=True, action="allow")