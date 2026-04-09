"""
FActScore: atomic-level factual precision evaluation.

This module implements a simplified LLM-based FActScore:
1. Rewrite Q+A into a self-contained declarative statement.
2. Decompose the statement into atomic claims.
3. Verify each claim against the provided documents using an LLM.
4. Compute the proportion of supported claims.
"""

import json
import logging
import re

from src.generation import generate
from src.evaluation.metrics import parse_structured_output

logger = logging.getLogger(__name__)


# ----------------------------- Prompt templates ----------------------------- #

REWRITE_SYSTEM = (
    "You are a precise text rewriting assistant. "
    "Given a question and its answer, rewrite them into one or more complete "
    "declarative sentences that are self-contained and factual. "
    "Do NOT add any information beyond what the question and answer provide."
)

REWRITE_USER = (
    "Rewrite the following question-answer pair into declarative sentence(s).\n\n"
    "Examples:\n"
    'Q: "In what year was the university founded?" A: "1755"\n'
    '→ The university was founded in 1755.\n\n'
    'Q: "Black Book starred the actress of what heritage?" A: "Dutch"\n'
    '→ The actress in Black Book is of Dutch heritage.\n\n'
    'Q: "Are Verdi and Thomas both opera composers?" A: "Yes"\n'
    '→ Both Verdi and Thomas are opera composers.\n\n'
    'Q: "Which genus contains more species?" A: "Greyia"\n'
    '→ Greyia contains more species.\n\n'
    "Q: \"{question}\" A: \"{answer}\"\n\n"
    "Output ONLY the declarative sentence(s), no other text."
)

DECOMPOSE_SYSTEM = (
    "You are a precise text analysis assistant. "
    "Your task is to decompose a statement into independent atomic claims. "
    "Each atomic claim should be a single, self-contained factual statement. "
    "If the statement contains only one fact, return exactly one claim."
)

DECOMPOSE_USER = (
    "Decompose the following statement into atomic claims. "
    "Return a JSON array of strings, one claim per element.\n\n"
    "Rules:\n"
    "- Each claim must be a standalone factual sentence.\n"
    "- If the statement is already atomic, return it as a single-element array.\n\n"
    "Statement: {statement}\n\n"
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
    "Is the claim supported by or consistent with the reference documents? "
    "A claim is \"supported\" if the documents contain information that confirms it, "
    "even if not using the exact same words. "
    "Answer with exactly one word: \"supported\" or \"unsupported\"."
)


# ------------------------------ Implementation ------------------------------ #

def rewrite_as_statement(question: str, answer: str, model_key: str) -> str:
    """
    Rewrite a Q+A pair into a self-contained declarative statement.
    """
    user_prompt = REWRITE_USER.format(question=question, answer=answer)
    raw = generate(
        model_key=model_key,
        system_prompt=REWRITE_SYSTEM,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=256,
    )
    return raw.strip()


def decompose_into_claims(statement: str, model_key: str) -> list[str]:
    """
    Decompose a declarative statement into a list of atomic factual claims.
    """
    if not statement or statement.strip() == "[ERROR]":
        return []

    user_prompt = DECOMPOSE_USER.format(statement=statement)
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
    question: str = "",
) -> dict:
    """
    Compute FActScore for a single answer against reference documents.

    Pipeline: Q+A → rewrite → declarative statement → decompose → atomic claims → verify.

    Returns:
        dict with keys: claims, num_claims, num_supported, factscore
    """
    # Extract just the answer content (strip Answer:/Citations: format)
    parsed = parse_structured_output(answer)
    answer_text = parsed["answer"]

    # Rewrite Q+A into declarative statement, then decompose
    if question:
        statement = rewrite_as_statement(question, answer_text, model_key)
    else:
        statement = answer_text

    claims = decompose_into_claims(statement, model_key)
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
