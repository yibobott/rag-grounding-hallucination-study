"""
FActScore: atomic-level factual precision evaluation.

This module implements a simplified LLM-based FActScore:
1. Decompose the answer into atomic claims.
2. Verify each claim against the provided documents using an LLM.
3. Compute the proportion of supported claims.
"""

import json
import logging
import re

from src.generation import generate

logger = logging.getLogger(__name__)


# ----------------------------- Prompt templates ----------------------------- #

DECOMPOSE_SYSTEM = (
    "You are a precise text analysis assistant. "
    "Your task is to decompose a given answer into independent atomic claims. "
    "Each atomic claim should be a single, self-contained factual statement."
)

DECOMPOSE_USER = (
    "Decompose the following answer into atomic claims. "
    "Return a JSON array of strings, one claim per element. "
    "Each claim must be a simple, standalone factual sentence.\n\n"
    "Answer: {answer}\n\n"
    "Output ONLY a JSON array, no other text."
)

VERIFY_SYSTEM = (
    "You are a rigorous fact-checking assistant. "
    "Given a claim and a set of reference documents, determine whether the claim "
    "is fully supported by the documents."
)

VERIFY_USER = (
    "Claim: {claim}\n\n"
    "Reference documents:\n{docs}\n\n"
    "Is the claim fully supported by the reference documents? "
    "Answer with exactly one word: \"supported\" or \"unsupported\"."
)


# ------------------------------ Implementation ------------------------------ #

def decompose_into_claims(answer: str, model_key: str) -> list[str]:
    """
    Decompose an answer into a list of atomic factual claims using an LLM.
    """
    if not answer or answer.strip() == "[ERROR]":
        return []

    user_prompt = DECOMPOSE_USER.format(answer=answer)
    raw = generate(
        model_key=model_key,
        system_prompt=DECOMPOSE_SYSTEM,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=512,
    )

    # Parse JSON array from the response
    try:
        # Try to extract a JSON array even if wrapped in markdown code block
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            claims = json.loads(match.group())
            if isinstance(claims, list):
                return [str(c).strip() for c in claims if str(c).strip()]
        return []
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to parse claims from LLM output: %s", e)
        return []


def verify_claim(
    claim: str,
    docs: list[dict],
    model_key: str,
) -> bool:
    """
    Verify whether a single claim is supported by the provided documents.
    """
    doc_str = "\n\n".join(
        f"[Doc {i+1}] {d['title']}: {d['text']}"
        for i, d in enumerate(docs)
    )
    user_prompt = VERIFY_USER.format(claim=claim, docs=doc_str)

    raw = generate(
        model_key=model_key,
        system_prompt=VERIFY_SYSTEM,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=16,
    )
    return "supported" in raw.lower() and "unsupported" not in raw.lower()


def factscore(
    answer: str,
    docs: list[dict],
    model_key: str,
) -> dict:
    """
    Compute FActScore for a single answer against reference documents.

    Returns:
        dict with keys: claims, num_claims, num_supported, factscore
    """
    claims = decompose_into_claims(answer, model_key)
    if not claims:
        return {
            "claims": [],
            "num_claims": 0,
            "num_supported": 0,
            "factscore": 0.0,
        }

    supported_flags = []
    for claim in claims:
        is_supported = verify_claim(claim, docs, model_key)
        supported_flags.append(is_supported)

    num_supported = sum(supported_flags)
    return {
        "claims": claims,
        "num_claims": len(claims),
        "num_supported": num_supported,
        "factscore": num_supported / len(claims),
    }
