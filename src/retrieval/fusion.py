"""
Reciprocal Rank Fusion (RRF) for hybrid retrieval.
"""

from src.retrieval.bm25 import bm25_retrieve
from src.retrieval.dense import dense_retrieve


def rrf_fusion(
    bm25_results: list[dict],
    dense_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Reciprocal Rank Fusion of two ranked lists.

    Args:
        bm25_results: list of dicts with 'original_index' (and optionally 'bm25_score')
        dense_results: list of dicts with 'original_index' (and optionally 'dense_score')
        k: RRF constant (usually 60)

    Returns:
        list of dicts (merged documents) sorted by RRF score descending,
        each dict contains keys from original documents plus:
            - 'rrf_score': float
            - 'bm25_rank': int or None
            - 'dense_rank': int or None
    """
    # Build rank maps: original_index -> rank (1-based)
    bm25_rank = {}
    for rank, doc in enumerate(bm25_results, start=1):
        bm25_rank[doc["original_index"]] = rank

    dense_rank = {}
    for rank, doc in enumerate(dense_results, start=1):
        dense_rank[doc["original_index"]] = rank

    # Collect all unique document indices from both lists
    all_indices = set(bm25_rank.keys()) | set(dense_rank.keys())

    # Compute RRF score for each document
    scores = {}
    for idx in all_indices:
        score = 0.0
        if idx in bm25_rank:
            score += 1.0 / (k + bm25_rank[idx])
        if idx in dense_rank:
            score += 1.0 / (k + dense_rank[idx])
        scores[idx] = score

    # Sort by score descending
    sorted_indices = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Build result list: need to recover document content.
    # We'll take the document dict from whichever list contains it (prefer BM25's copy)
    doc_map = {}
    for doc in bm25_results:
        doc_map[doc["original_index"]] = doc
    for doc in dense_results:
        if doc["original_index"] not in doc_map:
            doc_map[doc["original_index"]] = doc

    results = []
    for idx, rrf_score in sorted_indices:
        doc = dict(doc_map[idx])
        doc["rrf_score"] = rrf_score
        doc["bm25_rank"] = bm25_rank.get(idx)
        doc["dense_rank"] = dense_rank.get(idx)
        results.append(doc)

    return results


def hybrid_retrieve(
    query: str,
    docs: list[dict],
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[dict]:
    """
    Hybrid retrieval using BM25 + Contriever + RRF.

    Args:
        query: question string.
        docs: list of candidate documents.
        top_k: number of final documents to return.
        rrf_k: RRF constant.

    Returns:
        list[dict]: top-k documents after fusion.
    """
    # 1. Get BM25 results (rank all documents, or top enough)
    # We need the full ranking to compute RRF properly. Retrieve all docs.
    bm25_all = bm25_retrieve(query, docs, top_k=len(docs))
    dense_all = dense_retrieve(query, docs, top_k=len(docs))

    # 2. Fuse using RRF
    fused = rrf_fusion(bm25_all, dense_all, k=rrf_k)

    # 3. Return top_k
    return fused[:top_k]