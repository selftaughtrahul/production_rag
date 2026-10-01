"""Score the Tesla benchmark with this project's retriever and Claude.

Retrieval uses Qdrant hybrid search (dense + BM25 + RRF) and the cross-encoder
reranker. Answers use ClaudeService. Ragas and DeepEval then judge those answers.

    python files/evaluate_rag.py --user-id <user-id>
    python files/evaluate_rag.py --user-id <user-id> --limit 3
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import re
import sys
import types
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("evaluate_rag")


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


class RagEvaluator:
    """Hybrid retrieval plus Claude, matching the chat RAG path."""

    def __init__(self, user_id: str) -> None:
        from app.api.dependencies import get_cross_encoder_model, get_hybrid_retriever
        from app.core.config import Settings
        from app.core.device import resolve_torch_device
        from app.services.llm.claude import ClaudeService
        from app.services.retriever.reranker import CrossEncoderReranker

        if not user_id.strip():
            raise ValueError("user_id is required so retrieval stays on one tenant")
        self.user_id = user_id.strip()
        settings = Settings.from_environment()
        self.retriever = get_hybrid_retriever()
        self.reranker = CrossEncoderReranker(
            model=get_cross_encoder_model(
                settings.reranker_model,
                resolve_torch_device(settings.embedding_device),
            )
        )
        self.llm = ClaudeService()
        self._pool_k = max(settings.dense_top_k, 10)

    def retrieve(self, question: str, k: int) -> list[str]:
        """Return the top-k chunk texts from hybrid search plus reranking."""
        documents = self.retriever.retrieve(
            query=question,
            top_k=max(k, self._pool_k),
            metadata_filter={"user_id": self.user_id},
        )
        if not documents:
            return []
        ranked = self.reranker.rerank(query=question, documents=documents, top_k=k)
        return [document.text for document in ranked if document.text]

    def answer(self, question: str, chunks: list[str]) -> str:
        """Generate an answer with Claude from the retrieved chunks."""
        context = "\n\n".join(chunks)
        return self.llm.generate(question=question, context=context)


def _load_ragas() -> Any:
    """Import ragas. Stub a removed Vertex class that ragas still imports."""
    try:
        import ragas
    except ModuleNotFoundError as exc:
        if "vertexai" not in str(exc):
            raise
        stub = types.ModuleType("langchain_community.chat_models.vertexai")
        stub.ChatVertexAI = type("ChatVertexAI", (), {})  # type: ignore[attr-defined]
        sys.modules["langchain_community.chat_models.vertexai"] = stub
        import ragas
    return ragas


def score_ragas(cases: list[dict[str, Any]], model: str, api_key: str) -> dict[str, float]:
    """Ragas faithfulness, context precision, and context recall with Claude."""
    from anthropic import Anthropic

    _load_ragas()
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.llms import llm_factory
    from ragas.metrics.collections import context_precision, context_recall, faithfulness

    judge = llm_factory(model, provider="anthropic", client=Anthropic(api_key=api_key))
    samples = [
        SingleTurnSample(
            user_input=case["user_input"],
            retrieved_contexts=list(case["retrieved_contexts"]),
            response=case["response"],
            reference=case["reference"],
        )
        for case in cases
    ]
    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[faithfulness, context_precision, context_recall],
        llm=judge,
    )
    table = result.to_pandas()
    numeric = table.select_dtypes(include="number")
    summary = {column: float(numeric[column].mean()) for column in numeric.columns}
    logger.info("Ragas summary: %s", summary)
    return summary


def _deepeval_model(model: str) -> Any:
    try:
        from deepeval.models import AnthropicModel
    except ImportError:
        from deepeval.models.llms.anthropic_model import AnthropicModel
    try:
        return AnthropicModel(model=model)
    except TypeError:
        return AnthropicModel(model_name=model)


def score_deepeval(cases: list[dict[str, Any]], model: str) -> dict[str, float]:
    """DeepEval faithfulness, answer relevancy, and contextual precision."""
    try:
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            FaithfulnessMetric,
        )
        from deepeval.test_case import LLMTestCase
    except ImportError as exc:
        raise RuntimeError(
            "DeepEval is not installed. Install it with: pip install deepeval"
        ) from exc

    judge = _deepeval_model(model)
    metrics = {
        "deepeval_faithfulness": FaithfulnessMetric(
            threshold=0.5, model=judge, include_reason=False
        ),
        "deepeval_answer_relevancy": AnswerRelevancyMetric(
            threshold=0.5, model=judge, include_reason=False
        ),
        "deepeval_contextual_precision": ContextualPrecisionMetric(
            threshold=0.5, model=judge, include_reason=False
        ),
    }
    totals = {name: [] for name in metrics}
    for case in cases:
        test_case = LLMTestCase(
            input=case["user_input"],
            actual_output=case["response"],
            expected_output=case["reference"],
            retrieval_context=list(case["retrieved_contexts"]),
        )
        for name, metric in metrics.items():
            metric.measure(test_case)
            totals[name].append(float(metric.score or 0.0))
    summary = {
        name: (sum(values) / len(values) if values else 0.0)
        for name, values in totals.items()
    }
    logger.info("DeepEval summary: %s", summary)
    return summary


def _benchmark_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    beside_script = Path(__file__).resolve().parent / path
    if beside_script.is_file():
        return beside_script
    raise FileNotFoundError(f"Benchmark file not found: {path}")


def main(
    path: str = "tesla_q4_2024_rag_benchmark.jsonl",
    user_id: str = "",
    ks: tuple[int, ...] = (1, 3, 5, 10),
    run_generation: bool = True,
    run_ragas: bool = True,
    run_deepeval: bool = True,
    limit: int | None = None,
) -> dict[str, Any]:
    benchmark = _benchmark_path(path)
    items = [json.loads(line) for line in benchmark.read_text(encoding="utf-8").splitlines() if line.strip()]
    answerable = [item for item in items if item.get("answerable")]
    if limit is not None:
        answerable = answerable[:limit]

    pipeline = RagEvaluator(user_id)
    stats: dict[str, dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    cases: list[dict[str, Any]] = []

    for item in answerable:
        chunks = pipeline.retrieve(item["question"], max(ks))
        evidence = [norm(snippet) for snippet in item["evidence"]]
        rank = next(
            (
                index + 1
                for index, chunk in enumerate(chunks)
                if any(snippet in norm(chunk) for snippet in evidence)
            ),
            None,
        )
        for k in ks:
            top = norm(" ".join(chunks[:k]))
            stats[item["type"]][f"hit@{k}"].append(int(rank is not None and rank <= k))
            stats[item["type"]][f"recall@{k}"].append(
                sum(snippet in top for snippet in evidence) / len(evidence)
            )
        stats[item["type"]]["mrr"].append(1 / rank if rank else 0.0)
        if run_generation:
            prediction = pipeline.answer(item["question"], chunks[:5])
            cases.append(
                {
                    "id": item["id"],
                    "user_input": item["question"],
                    "retrieved_contexts": chunks[:5],
                    "reference": item["ground_truth"],
                    "response": prediction,
                }
            )
            logger.info("answered %s", item["id"])

    report: dict[str, Any] = {"retrieval": {}}
    for question_type, metrics in stats.items():
        print(f"\n[{question_type}]")
        report["retrieval"][question_type] = {}
        for name, values in metrics.items():
            mean = sum(values) / len(values)
            report["retrieval"][question_type][name] = mean
            print(f"  {name:10s} {mean:.3f}  (n={len(values)})")

    if cases:
        predictions_path = benchmark.with_name("predictions.json")
        predictions_path.write_text(json.dumps(cases, indent=2), encoding="utf-8")
        print(f"\nSaved {predictions_path}")

    if cases and (run_ragas or run_deepeval):
        if not pipeline.llm.api_key.strip():
            raise RuntimeError("ANTHROPIC_API_KEY is required for Ragas and DeepEval")
        if run_ragas:
            report["ragas"] = score_ragas(cases, pipeline.llm.model, pipeline.llm.api_key)
            print("\n[ragas]")
            for name, value in report["ragas"].items():
                print(f"  {name:28s} {value:.3f}")
        if run_deepeval:
            report["deepeval"] = score_deepeval(cases, pipeline.llm.model)
            print("\n[deepeval]")
            for name, value in report["deepeval"].items():
                print(f"  {name:28s} {value:.3f}")
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate hybrid RAG on the Tesla benchmark.")
    parser.add_argument("--user-id", required=True, help="Tenant id that owns the indexed Tesla document.")
    parser.add_argument("--path", default="tesla_q4_2024_rag_benchmark.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N answerable questions.")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip Claude, Ragas, and DeepEval.")
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--skip-deepeval", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    main(
        path=args.path,
        user_id=args.user_id,
        run_generation=not args.retrieval_only,
        run_ragas=not args.retrieval_only and not args.skip_ragas,
        run_deepeval=not args.retrieval_only and not args.skip_deepeval,
        limit=args.limit,
    )
