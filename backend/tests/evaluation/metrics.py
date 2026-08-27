"""
Evaluation framework for Enterprise-RAG.

Provides:
  - Golden dataset format (domain-independent)
  - Retrieval evaluation (Recall@K, Precision@K, MRR, NDCG@K)
  - Generation evaluation (groundedness, answer relevance, citation correctness)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── Golden Dataset Format ────────────────────────────────────────────────────────

@dataclass
class GoldenEntry:
    """
    One entry in the evaluation golden set.
    Domain-independent; expected fields represent ground truth.
    """
    question: str
    expected_document: str          # filename or document ID
    expected_page: int | None = None
    expected_section: str | None = None
    expected_evidence: str = ""     # key passage that must appear in context
    expected_answer: str = ""       # canonical answer text


def load_golden_dataset(path: str | Path) -> list[GoldenEntry]:
    """Load golden entries from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [GoldenEntry(**item) for item in data]


# ── Retrieval Metrics ────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    chunk_id: str
    document_id: str
    page_number: int | None = None
    score: float = 0.0
    content: str = ""


@dataclass
class RetrievalMetrics:
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    k: int = 10
    total_queries: int = 0
    hit_count: int = 0


def _is_relevant(result: RetrievalResult, entry: GoldenEntry) -> bool:
    """Determine if a retrieval result is relevant to the golden entry."""
    # Document-level relevance
    if entry.expected_document not in (result.document_id, result.chunk_id):
        return False
    # Page-level match when specified
    if entry.expected_page is not None and result.page_number != entry.expected_page:
        return False
    # Evidence-level match when specified
    if entry.expected_evidence and entry.expected_evidence.lower() not in result.content.lower():
        return False
    return True


def evaluate_retrieval(
    queries: list[GoldenEntry],
    retrieved: list[list[RetrievalResult]],
    k: int = 10,
) -> RetrievalMetrics:
    """
    Compute Recall@K, Precision@K, MRR, NDCG@K across the golden dataset.

    Args:
        queries:   Golden dataset entries.
        retrieved: Per-query list of retrieved results (same order as queries).
        k:         Cutoff depth.
    """
    assert len(queries) == len(retrieved), "Queries and retrieved lists must match"

    total = len(queries)
    recall_sum = precision_sum = mrr_sum = ndcg_sum = 0.0
    hit_count = 0

    for entry, results in zip(queries, retrieved):
        top_k = results[:k]
        relevant_flags = [_is_relevant(r, entry) for r in top_k]
        num_relevant = sum(relevant_flags)

        # Recall@K (binary relevant = 1 expected doc)
        recall = 1.0 if num_relevant > 0 else 0.0
        recall_sum += recall
        if recall > 0:
            hit_count += 1

        # Precision@K
        precision_sum += num_relevant / k

        # MRR
        for rank, flag in enumerate(relevant_flags, start=1):
            if flag:
                mrr_sum += 1.0 / rank
                break

        # NDCG@K
        dcg = sum(1.0 / math.log2(rank + 1) for rank, flag in enumerate(relevant_flags, start=1) if flag)
        ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(num_relevant + 1, k + 1)))
        ndcg_sum += (dcg / ideal_dcg) if ideal_dcg > 0 else 0.0

    return RetrievalMetrics(
        recall_at_k=recall_sum / total,
        precision_at_k=precision_sum / total,
        mrr=mrr_sum / total,
        ndcg_at_k=ndcg_sum / total,
        k=k,
        total_queries=total,
        hit_count=hit_count,
    )


# ── Generation Metrics ────────────────────────────────────────────────────────────

@dataclass
class GenerationMetrics:
    groundedness_rate: float = 0.0    # fraction of answers supported by evidence
    answer_relevance: float = 0.0     # fraction of answers that address the question
    citation_correctness: float = 0.0 # fraction of citations pointing to correct doc/page
    unsupported_claim_rate: float = 0.0
    total_queries: int = 0


def evaluate_generation(
    queries: list[GoldenEntry],
    answers: list[str],
    citations: list[list[dict]],
    contexts: list[str],
) -> GenerationMetrics:
    """
    Evaluate generation quality with simple heuristic metrics.
    NOTE: These are objective heuristics only. Semantic LLM-based scoring must be
    enabled explicitly and requires an additional LLM call.

    Args:
        queries:   Golden entries.
        answers:   Generated answers (same order).
        citations: Per-query list of citation dicts {document, page, ...}.
        contexts:  Retrieved context strings used to generate each answer.
    """
    assert len(queries) == len(answers), "Mismatch between queries and answers"
    total = len(queries)

    grounded_count = 0
    relevant_count = 0
    correct_citation_count = 0
    unsupported_count = 0

    for entry, answer, cited, context in zip(queries, answers, citations, contexts):
        # Groundedness: check if any expected evidence appears in context
        is_grounded = bool(context and entry.expected_evidence and
                           entry.expected_evidence.lower() in context.lower())
        if is_grounded:
            grounded_count += 1
        else:
            unsupported_count += 1

        # Relevance: answer is non-empty and contains at least one token from expected
        exp_tokens = set(entry.expected_answer.lower().split())
        ans_tokens = set(answer.lower().split())
        overlap = len(exp_tokens & ans_tokens)
        if answer and exp_tokens and overlap / len(exp_tokens) > 0.3:
            relevant_count += 1

        # Citation correctness: at least one citation matches expected document
        for cit in cited:
            if entry.expected_document in (cit.get("document_id", ""), cit.get("filename", "")):
                correct_citation_count += 1
                break

    return GenerationMetrics(
        groundedness_rate=grounded_count / total,
        answer_relevance=relevant_count / total,
        citation_correctness=correct_citation_count / total,
        unsupported_claim_rate=unsupported_count / total,
        total_queries=total,
    )


# ── Report Output ─────────────────────────────────────────────────────────────────

def print_retrieval_report(metrics: RetrievalMetrics, label: str = "") -> None:
    tag = f"[{label}] " if label else ""
    print(f"\n{tag}── Retrieval Metrics (k={metrics.k}) ──")
    print(f"  Recall@{metrics.k}:    {metrics.recall_at_k:.4f}")
    print(f"  Precision@{metrics.k}: {metrics.precision_at_k:.4f}")
    print(f"  MRR:           {metrics.mrr:.4f}")
    print(f"  NDCG@{metrics.k}:      {metrics.ndcg_at_k:.4f}")
    print(f"  Hits:          {metrics.hit_count}/{metrics.total_queries}")


def print_generation_report(metrics: GenerationMetrics, label: str = "") -> None:
    tag = f"[{label}] " if label else ""
    print(f"\n{tag}── Generation Metrics ──")
    print(f"  Groundedness Rate:    {metrics.groundedness_rate:.4f}")
    print(f"  Answer Relevance:     {metrics.answer_relevance:.4f}")
    print(f"  Citation Correctness: {metrics.citation_correctness:.4f}")
    print(f"  Unsupported Rate:     {metrics.unsupported_claim_rate:.4f}")
    print(f"  Total Queries:        {metrics.total_queries}")
