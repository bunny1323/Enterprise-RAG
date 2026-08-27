"""
Evaluation runner for Enterprise-RAG.

Usage:
  python tests/evaluation/run_eval.py --golden tests/evaluation/golden_dataset.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Allow running directly from the backend directory
sys.path.insert(0, str(Path(__file__).parents[2]))

from tests.evaluation.metrics import (
    GoldenEntry,
    RetrievalResult,
    evaluate_generation,
    evaluate_retrieval,
    load_golden_dataset,
    print_generation_report,
    print_retrieval_report,
)


def _mock_retrieval(entry: GoldenEntry, k: int = 10) -> list[RetrievalResult]:
    """
    Placeholder: replace this with a real call to the RAG retrieval pipeline.
    Returns empty results (zero recall) until connected to a real backend.
    """
    return []


def _mock_generation(entry: GoldenEntry, context: str) -> tuple[str, list[dict]]:
    """
    Placeholder: replace with real LLM generation call.
    """
    return "", []


async def run_retrieval_evaluation(
    golden: list[GoldenEntry],
    k: int = 10,
    label: str = "Current",
) -> None:
    retrieved = [_mock_retrieval(e, k) for e in golden]
    metrics = evaluate_retrieval(golden, retrieved, k=k)
    print_retrieval_report(metrics, label=label)


async def run_generation_evaluation(golden: list[GoldenEntry]) -> None:
    answers = []
    citations = []
    contexts = []

    for entry in golden:
        results = _mock_retrieval(entry)
        context = " ".join(r.content for r in results)
        answer, cited = _mock_generation(entry, context)
        answers.append(answer)
        citations.append(cited)
        contexts.append(context)

    metrics = evaluate_generation(golden, answers, citations, contexts)
    print_generation_report(metrics, label="Current")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enterprise-RAG Evaluation Runner")
    parser.add_argument("--golden", required=True, help="Path to golden dataset JSON")
    parser.add_argument("--k", type=int, default=10, help="Cutoff k for retrieval metrics")
    parser.add_argument("--retrieval", action="store_true", help="Run retrieval evaluation")
    parser.add_argument("--generation", action="store_true", help="Run generation evaluation")
    args = parser.parse_args()

    golden = load_golden_dataset(args.golden)
    print(f"Loaded {len(golden)} golden entries from {args.golden}")

    run_all = not args.retrieval and not args.generation

    if args.retrieval or run_all:
        asyncio.run(run_retrieval_evaluation(golden, k=args.k))

    if args.generation or run_all:
        asyncio.run(run_generation_evaluation(golden))


if __name__ == "__main__":
    main()
