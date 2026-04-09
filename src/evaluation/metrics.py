"""
Answer-level and retrieval-level evaluation metrics.
"""

import re
import string
from collections import Counter

import numpy as np


# ---------------------- Structured output parsing ---------------------- #

def parse_structured_output(raw: str) -> dict:
    """
    Parse structured LLM output into answer and citations components.

    Expected format:
        Answer: <short answer>
        Citations: [Doc 1][Doc 2] ...

    Returns:
        {"answer": str, "citations_line": str, "raw": str}
        If parsing fails, "answer" falls back to the full raw text
        with [Doc N] tags stripped.
    """
    answer_match = re.search(
        r"(?i)^answer\s*:\s*(.+?)(?:\n|$)", raw, re.MULTILINE,
    )
    citations_match = re.search(
        r"(?i)^citations?\s*:\s*(.+?)(?:\n|$)", raw, re.MULTILINE,
    )

    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        # Fallback: strip [Doc N] tags from raw output
        answer = re.sub(r"\[Doc\s*\d+\]", "", raw).strip()

    citations_line = citations_match.group(1).strip() if citations_match else ""

    return {"answer": answer, "citations_line": citations_line, "raw": raw}


# ---------------------------- Text normalization ---------------------------- #

def normalize_answer(s: str) -> str:
    """
    Lower-case, remove articles / punctuation / extra whitespace.
    """
    s = s.lower()
    # remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove punctuation
    s = "".join(ch for ch in s if ch not in string.punctuation)
    # collapse whitespace
    s = " ".join(s.split())
    return s


# --------------------------- Answer-level metrics --------------------------- #

def exact_match(prediction: str, gold: str) -> float:
    """
    Binary exact-match after normalization.
    """
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    """
    Token-level F1 between prediction and gold answer.
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# -------------------- Semantic match (lazy-loaded model) -------------------- #

_st_model = None


def _get_st_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model


def semantic_match(prediction: str, gold: str) -> float:
    """
    Cosine similarity between sentence embeddings.
    """
    model = _get_st_model()
    embs = model.encode([prediction, gold], convert_to_numpy=True)
    cos = np.dot(embs[0], embs[1]) / (
        np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]) + 1e-9
    )
    return float(cos)


# -------------------------- Retrieval-level metric -------------------------- #

def retrieval_precision_at_k(retrieved_titles: list[str],
                              gold_titles: set[str],
                              k: int = 5) -> float:
    """
    Precision@K: fraction of top-K retrieved docs that are gold.
    """
    top_k = retrieved_titles[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for t in top_k if t in gold_titles)
    return hits / len(top_k)


# ---------------------------- Aggregation helper ---------------------------- #

def aggregate_metrics(all_metrics: list[dict]) -> dict:
    """
    Average per-example metric dicts into aggregate scores.
    """
    if not all_metrics:
        return {}
    keys = all_metrics[0].keys()
    agg = {}
    for k in keys:
        vals = [m[k] for m in all_metrics if k in m]
        agg[f"mean_{k}"] = float(np.mean(vals))
    agg["n"] = len(all_metrics)
    return agg
