"""
FActScore: atomic-level factual precision evaluation.

This module implements a simplified LLM-based FActScore:
1. Decompose the answer into atomic claims.
2. Verify each claim against the provided documents using an LLM.
3. Compute the proportion of supported claims.

To be implemented when E-Oracle and E0 baseline are complete.
"""

# TODO: Implement decompose_into_claims(answer, model_key) -> list[str]
# TODO: Implement verify_claim(claim, docs, model_key) -> bool
# TODO: Implement factscore(answer, docs, model_key) -> float
