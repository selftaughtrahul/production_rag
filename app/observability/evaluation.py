from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class EvaluationResult:
    """
    Result of evaluating a RAG response.
    """

    question: str
    answer: str
    context: str
    answer_present: bool
    context_present: bool
    passed: bool
    details: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "answer_present": self.answer_present,
            "context_present": self.context_present,
            "details": self.details,
        }


class RAGEvaluator:
    """
    Basic evaluator for RAG responses.

    This is intentionally simple.

    Later this can be extended with:
    - LLM-as-a-judge
    - Faithfulness
    - Context relevance
    - Answer relevance
    - Correctness
    """

    def evaluate(self,question: str,answer: str,context: str,) -> EvaluationResult:

        answer_present = bool(answer and answer.strip())
        context_present = bool(context and context.strip())

        passed = (answer_present and context_present)

        details = {"answer_length": len(answer.strip()) if answer else 0,"context_length": len(context.strip() ) if context else 0}

        return EvaluationResult(
            question=question,
            answer=answer,
            context=context,
            answer_present=answer_present,
            context_present=context_present,
            passed=passed,
            details=details,
        )
