from app.guardrails.base import BaseOutputGuardrail, GuardrailResult

_REFUSAL_MARKERS = (
    "don't have enough information",
    "do not have enough information",
    "i don't know",
)


def _tokens(text: str) -> set[str]:
    words = [w.lower() for w in (text or "").split() if len(w) > 2]
    return set(words)


class TokenOverlapGroundingEvaluator:
    """Lexical overlap between the answer and retrieved context. No extra model."""

    async def evaluate(self, response: str, context: str) -> float:
        lowered = (response or "").lower()
        if any(marker in lowered for marker in _REFUSAL_MARKERS):
            return 1.0
        answer = _tokens(response)
        ctx = _tokens(context)
        if not answer:
            return 1.0
        if not ctx:
            return 0.0
        return len(answer & ctx) / len(answer)


class GroundingGuardrail(BaseOutputGuardrail):
    def __init__(self, evaluator=None, threshold: float = 0.15):
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