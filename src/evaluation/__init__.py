"""
Evaluation — unified interface combining metrics and citation modules.
"""

from src.evaluation.metrics import (
    parse_structured_output,
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

    The raw prediction is parsed into structured components:
    - "Answer: ..." → used for EM / F1 / Semantic Match (answer accuracy)
    - Full raw output → used for citation grounding / hallucination / FActScore

    Args:
        prediction: raw LLM output (may contain Answer:/Citations: format).
        question: the original question (used for hallucination/faithfulness).
        model_key: required for LLM-based metrics.
        compute_hallucination: if True, run hallucination_check + faithfulness_score.
        compute_factscore: if True, run atomic FActScore (subset only).
    """
    parsed = parse_structured_output(prediction)
    answer = parsed["answer"]

    # ---- Answer accuracy (use extracted short answer) ---- #
    metrics: dict = {
        "em": exact_match(answer, gold_answer),
        "token_f1": token_f1(answer, gold_answer),
        "semantic_match": semantic_match(answer, gold_answer),
    }

    if retrieved_titles is not None and gold_titles is not None:
        metrics["retrieval_precision_at_5"] = retrieval_precision_at_k(
            retrieved_titles, gold_titles, k=5
        )

    # ---- Citation grounding (use full raw output) ---- #
    if docs is not None and gold_titles is not None:
        cg = citation_grounding_rate(prediction, docs, gold_titles)
        metrics["citation_grounding_rate"] = cg["grounding_rate"]
        metrics["num_citations"] = cg["num_citations"]
        metrics["num_grounded"] = cg["num_grounded"]

    # ---- Lightweight LLM-based hallucination metrics (full raw output) ---- #
    if compute_hallucination and docs is not None and model_key is not None:
        metrics["has_hallucination"] = hallucination_check(
            prediction, docs, model_key, question=question,
        )
        metrics["faithfulness"] = faithfulness_score(
            prediction, docs, model_key, question=question,
        )

    # ---- Atomic FActScore (subset only, extracted answer) ---- #
    if compute_factscore and docs is not None and model_key is not None:
        fs = factscore(prediction, docs, model_key, question=question)
        metrics["factscore"] = fs["factscore"]
        metrics["num_claims"] = fs["num_claims"]
        metrics["num_supported_claims"] = fs["num_supported"]

    return metrics


__all__ = [
    "parse_structured_output",
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
