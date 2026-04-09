"""Evaluation — unified interface combining metrics and citation modules."""

from src.evaluation.metrics import (
    normalize_answer,
    exact_match,
    token_f1,
    semantic_match,
    retrieval_precision_at_k,
    aggregate_metrics,
)
from src.evaluation.citation import (
    extract_citations,
    citation_grounding_rate,
)


def compute_all_metrics(
    prediction: str,
    gold_answer: str,
    docs: list[dict] | None = None,
    gold_titles: set[str] | None = None,
    retrieved_titles: list[str] | None = None,
) -> dict:
    """Compute all applicable metrics for a single example."""
    metrics: dict = {
        "em": exact_match(prediction, gold_answer),
        "token_f1": token_f1(prediction, gold_answer),
        "semantic_match": semantic_match(prediction, gold_answer),
    }

    if retrieved_titles is not None and gold_titles is not None:
        metrics["retrieval_precision_at_5"] = retrieval_precision_at_k(
            retrieved_titles, gold_titles, k=5
        )

    if docs is not None and gold_titles is not None:
        cg = citation_grounding_rate(prediction, docs, gold_titles)
        metrics["citation_grounding_rate"] = cg["grounding_rate"]
        metrics["num_citations"] = cg["num_citations"]
        metrics["num_grounded"] = cg["num_grounded"]

    return metrics


__all__ = [
    "normalize_answer",
    "exact_match",
    "token_f1",
    "semantic_match",
    "retrieval_precision_at_k",
    "aggregate_metrics",
    "extract_citations",
    "citation_grounding_rate",
    "compute_all_metrics",
]
