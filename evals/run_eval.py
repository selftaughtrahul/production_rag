"""Golden-set eval harness for the RAG pipeline.

Offline (CI, no API key):

    python evals/run_eval.py --offline

Ragas LLM-as-judge when ANTHROPIC_API_KEY is set:

    python evals/run_eval.py --with-ragas
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
logger = logging.getLogger("evals")

_REQUIRED_FIELDS = (
    "id",
    "category",
    "user_input",
    "retrieved_contexts",
    "reference",
    "response",
)


def _tokens(text: str) -> set[str]:
    return {part.lower() for part in (text or "").split() if len(part) > 2}


def load_golden(path: Path | None = None) -> dict[str, Any]:
    golden_path = path or (ROOT / "golden.json")
    payload = json.loads(golden_path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("golden.json must contain a non-empty cases list")
    for case in cases:
        missing = [field for field in _REQUIRED_FIELDS if field not in case]
        if missing:
            raise ValueError(f"case {case.get('id')!r} missing fields: {missing}")
        if not isinstance(case["retrieved_contexts"], list):
            raise ValueError(f"case {case['id']!r} retrieved_contexts must be a list")
    return payload


def load_baseline(path: Path | None = None) -> dict[str, Any]:
    baseline_path = path or (ROOT / "baseline.json")
    return json.loads(baseline_path.read_text(encoding="utf-8"))


def context_hit(case: dict[str, Any]) -> float | None:
    """Fraction of reference tokens found in retrieved contexts. None if no contexts."""
    contexts = case["retrieved_contexts"]
    if not contexts:
        return None
    ref = _tokens(case["reference"])
    if not ref:
        return None
    blob = _tokens(" ".join(str(chunk) for chunk in contexts))
    return len(ref & blob) / len(ref)


def answer_overlap(case: dict[str, Any]) -> float:
    ref = _tokens(case["reference"])
    ans = _tokens(case["response"])
    if not ref or not ans:
        return 0.0
    return len(ref & ans) / len(ref)


def score_offline(cases: list[dict[str, Any]]) -> dict[str, float]:
    hits = [value for value in (context_hit(case) for case in cases) if value is not None]
    overlaps = [answer_overlap(case) for case in cases]
    return {
        "n_cases": float(len(cases)),
        "context_hit": sum(hits) / len(hits) if hits else 0.0,
        "answer_overlap": sum(overlaps) / len(overlaps) if overlaps else 0.0,
    }


def assert_baseline(scores: dict[str, float], baseline: dict[str, Any]) -> None:
    if scores["n_cases"] < float(baseline["min_cases"]):
        raise SystemExit(
            f"too few golden cases: {scores['n_cases']} < {baseline['min_cases']}"
        )
    if scores["context_hit"] < float(baseline["min_context_hit"]):
        raise SystemExit(
            f"context_hit {scores['context_hit']:.3f} < {baseline['min_context_hit']}"
        )
    if scores["answer_overlap"] < float(baseline["min_answer_overlap"]):
        raise SystemExit(
            f"answer_overlap {scores['answer_overlap']:.3f} < {baseline['min_answer_overlap']}"
        )


def run_ragas(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """LLM-judge metrics. Requires a working ragas install and ANTHROPIC_API_KEY."""
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    samples = [
        SingleTurnSample(
            user_input=case["user_input"],
            retrieved_contexts=list(case["retrieved_contexts"]),
            response=case["response"],
            reference=case["reference"],
        )
        for case in cases
    ]
    dataset = EvaluationDataset(samples=samples)
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )
    table = result.to_pandas()
    numeric = table.select_dtypes(include="number")
    summary = {column: float(numeric[column].mean()) for column in numeric.columns}
    logger.info("Ragas summary: %s", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run the RAG golden-set eval.")
    parser.add_argument("--offline", action="store_true", help="Lexical metrics only (CI).")
    parser.add_argument("--with-ragas", action="store_true", help="Call ragas.evaluate.")
    args = parser.parse_args(argv)

    if not args.offline and not args.with_ragas:
        args.offline = True

    golden = load_golden()
    baseline = load_baseline()
    cases: list[dict[str, Any]] = golden["cases"]
    scores = score_offline(cases)
    logger.info(
        "offline golden=%s cases=%s context_hit=%.3f answer_overlap=%.3f",
        golden.get("version"),
        int(scores["n_cases"]),
        scores["context_hit"],
        scores["answer_overlap"],
    )
    assert_baseline(scores, baseline)

    if args.with_ragas:
        if not os.getenv("ANTHROPIC_API_KEY", "").strip():
            logger.warning("ANTHROPIC_API_KEY unset; skipping ragas.evaluate")
            return 0
        try:
            run_ragas(cases)
        except Exception:
            logger.exception("ragas.evaluate failed")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
