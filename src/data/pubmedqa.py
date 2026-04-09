"""
PubMedQA data loading (used in E6 cross-domain evaluation).
"""

import random

from datasets import load_dataset


def load_pubmedqa(sample_size: int = 500, seed: int = 42, cache_dir: str | None = None):
    """
    Load PubMedQA labeled subset and sample.

    Returns:
        list[dict]: Each dict has keys:
            id, question, answer, long_answer, context
    """
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train",
                       cache_dir=cache_dir)

    random.seed(seed)
    indices = sorted(random.sample(range(len(ds)), min(sample_size, len(ds))))
    subset = ds.select(indices)

    samples = []
    for row in subset:
        samples.append({
            "id": str(row["pubid"]),
            "question": row["question"],
            "answer": row["final_decision"],          # yes / no / maybe
            "long_answer": row["long_answer"],
            "context": row["context"],
        })
    return samples
