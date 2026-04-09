"""
Lightweight LLM-based hallucination detection and faithfulness scoring.

These metrics each require only ONE LLM call per example, making them
suitable for full-scale evaluation (e.g., 500 samples).

- hallucination_check: binary YES/NO — does the answer contain hallucination?
- faithfulness_score: continuous 0–1 — how faithful is the answer to the docs?
"""

import logging
import re

from src.generation import generate

logger = logging.getLogger(__name__)


# ----------------------------- Prompt templates ----------------------------- #

HALLUCINATION_SYSTEM = (
    "You are an expert hallucination detector. "
    "Given a question, reference documents, and a generated answer, determine "
    "whether the answer contains any hallucinated information — i.e., claims "
    "that are NOT supported by or contradict the reference documents."
)

HALLUCINATION_USER = (
    "Question: {question}\n\n"
    "Reference documents:\n{docs}\n\n"
    "Generated answer: {answer}\n\n"
    "Does the answer contain any hallucinated information (claims not supported "
    "by the reference documents)?\n"
    "Answer with exactly one word: \"YES\" or \"NO\"."
)

FAITHFULNESS_SYSTEM = (
    "You are an expert faithfulness evaluator. "
    "Given a question, reference documents, and a generated answer, rate how "
    "faithful the answer is to the reference documents on a scale from 0.0 to 1.0.\n"
    "- 1.0: every claim in the answer is fully supported by the documents.\n"
    "- 0.5: some claims are supported, some are not.\n"
    "- 0.0: the answer is entirely unsupported or contradicts the documents."
)

FAITHFULNESS_USER = (
    "Question: {question}\n\n"
    "Reference documents:\n{docs}\n\n"
    "Generated answer: {answer}\n\n"
    "Rate the faithfulness of the answer to the reference documents.\n"
    "Output ONLY a single number between 0.0 and 1.0, nothing else."
)


# ------------------------------ Implementation ------------------------------ #

def _format_docs_plain(docs: list[dict]) -> str:
    """
    Format documents for hallucination / faithfulness prompts.
    """
    return "\n\n".join(
        f"[Doc {i+1}] {d['title']}: {d['text']}"
        for i, d in enumerate(docs)
    )


def hallucination_check(
    answer: str,
    docs: list[dict],
    model_key: str,
    question: str = "",
) -> bool:
    """
    Binary hallucination detection via LLM.

    Returns:
        True if the answer contains hallucination, False otherwise.
    """
    if not answer or answer.strip() == "[ERROR]":
        return True

    doc_str = _format_docs_plain(docs)
    user_prompt = HALLUCINATION_USER.format(
        question=question, docs=doc_str, answer=answer,
    )

    raw = generate(
        model_key=model_key,
        system_prompt=HALLUCINATION_SYSTEM,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=8,
    )
    return raw.strip().upper().startswith("YES")


def faithfulness_score(
    answer: str,
    docs: list[dict],
    model_key: str,
    question: str = "",
) -> float:
    """
    Continuous faithfulness score (0.0–1.0) via LLM.

    Returns:
        float between 0.0 and 1.0.
    """
    if not answer or answer.strip() == "[ERROR]":
        return 0.0

    doc_str = _format_docs_plain(docs)
    user_prompt = FAITHFULNESS_USER.format(
        question=question, docs=doc_str, answer=answer,
    )

    raw = generate(
        model_key=model_key,
        system_prompt=FAITHFULNESS_SYSTEM,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=8,
    )

    # Extract a float from the response
    match = re.search(r"([01](?:\.\d+)?)", raw.strip())
    if match:
        score = float(match.group(1))
        return max(0.0, min(1.0, score))

    logger.warning("Failed to parse faithfulness score from: %s", raw)
    return 0.0
