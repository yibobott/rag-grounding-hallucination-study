"""
BM25 sparse retrieval for HotpotQA distractor setting.

Given a query and a list of candidate documents, rank documents by BM25 score
and return the top-k most relevant ones.
"""

import numpy as np
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    """
    Simple whitespace tokenization with lowercasing.
    """
    return text.lower().split()


def bm25_retrieve(
    query: str,
    docs: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Rank candidate documents by BM25 relevance to the query.

    Args:
        query: the question string.
        docs: list of {"title": str, "text": str, ...} candidate documents.
        top_k: number of documents to return.

    Returns:
        list[dict]: top-k documents sorted by BM25 score (highest first).
        Each dict is a copy of the original doc with added keys:
            - "bm25_score": float
            - "original_index": int (0-based index in the input list)
    """
    if not docs:
        return []

    # Tokenize each document (title + text for richer signal)
    corpus = [_tokenize(f"{d['title']} {d['text']}") for d in docs]
    bm25 = BM25Okapi(corpus)

    # Score the query against all documents
    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens)

    # Rank by score descending, take top-k
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        doc = dict(docs[idx])  # shallow copy
        doc["bm25_score"] = float(scores[idx])
        doc["original_index"] = int(idx)
        results.append(doc)

    return results
