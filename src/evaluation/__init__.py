"""
Evaluation — unified interface combining metrics and citation modules.
"""

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
from src.evaluation.factscore import (
    decompose_into_claims,
    verify_claim,
    factscore,
)
from src.evaluation.hallucination import (
    hallucination_check,
    faithfulness_score,
)


def compute_all_metrics(
    prediction: str,
    gold_answer: str,
    question: str = "",
    docs: list[dict] | None = None,
    gold_titles: set[str] | None = None,
    retrieved_titles: list[str] | None = None,
    model_key: str | None = None,
    compute_hallucination: bool = False,
    compute_factscore: bool = False,
) -> dict:
    """
    Compute all applicable metrics for a single example.

    Args:
        question: the original question (used for hallucination/faithfulness).
        model_key: required for LLM-based metrics (hallucination, faithfulness, FActScore).
        compute_hallucination: if True, run hallucination_check + faithfulness_score (1+1 LLM calls).
        compute_factscore: if True, run atomic FActScore (N+1 LLM calls, use for subset only).
    """
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

    # ---- Lightweight LLM-based hallucination metrics (full-scale) ---- #
    if compute_hallucination and docs is not None and model_key is not None:
        metrics["has_hallucination"] = hallucination_check(
            prediction, docs, model_key, question=question,
        )
        metrics["faithfulness"] = faithfulness_score(
            prediction, docs, model_key, question=question,
        )

    # ---- Atomic FActScore (subset only — expensive) ---- #
    if compute_factscore and docs is not None and model_key is not None:
        fs = factscore(prediction, docs, model_key)
        metrics["factscore"] = fs["factscore"]
        metrics["num_claims"] = fs["num_claims"]
        metrics["num_supported_claims"] = fs["num_supported"]

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
    "hallucination_check",
    "faithfulness_score",
    "decompose_into_claims",
    "verify_claim",
    "factscore",
    "compute_all_metrics",
]
