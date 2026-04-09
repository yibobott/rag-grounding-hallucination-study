"""
HotpotQA data loading and context extraction.

On first call, load_hotpotqa() downloads data from HuggingFace, samples
with seed=42, and caches the result as a local JSON file under data/.
Subsequent calls read directly from the local file (no network needed).
"""

import json
import logging
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

logger = logging.getLogger(__name__)

# Local cache directory (project_root/data/)
_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def _cache_path(sample_size: int, seed: int) -> Path:
    """
    Return the path for the local cache file, e.g. data/hotpotqa_500_seed42.json
    """
    return _DATA_DIR / f"hotpotqa_{sample_size}_seed{seed}.json"


def load_hotpotqa(sample_size: int = 500, seed: int = 42, cache_dir: str | None = None):
    """
    Load HotpotQA dev set (distractor) and return a fixed-size sample.

    On first call, downloads from HuggingFace and saves to a local JSON file.
    On subsequent calls, loads directly from the local file (fast, offline).

    Returns:
        list[dict]: Each dict has keys:
            id, question, answer, type, level, supporting_facts, context
    """
    local_file = _cache_path(sample_size, seed)

    # Fast path: load from local cache
    if local_file.exists():
        logger.info("Loading HotpotQA from local cache: %s", local_file)
        with open(local_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # Slow path: download from HuggingFace, sample, and cache
    logger.info("Local cache not found. Downloading HotpotQA from HuggingFace...")
    ds = load_dataset("hotpot_qa", "distractor", split="validation",
                       cache_dir=cache_dir)

    random.seed(seed)
    indices = sorted(random.sample(range(len(ds)), min(sample_size, len(ds))))
    subset = ds.select(indices)

    samples = []
    for row in subset:
        samples.append({
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "type": row["type"],
            "level": row["level"],
            "supporting_facts": row["supporting_facts"],
            "context": row["context"],
        })

    # Save to local cache
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(local_file, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False)
    logger.info("Cached %d samples to %s", len(samples), local_file)

    return samples


def extract_oracle_docs(example: dict) -> list[dict]:
    """
    Extract gold supporting *passages* (full paragraphs) for a single example.

    For each title mentioned in ``supporting_facts``, return the complete
    paragraph (all sentences) from ``context``.  This is the document-level
    Oracle used in E-Oracle.

    Returns:
        list[dict]: [{"title": str, "text": str, "sentences": list[str]}, ...]
    """
    sf = example["supporting_facts"]
    ctx = example["context"]

    # Build lookup: title -> list[str]
    title_to_sents: dict[str, list[str]] = {}
    for title, sents in zip(ctx["title"], ctx["sentences"]):
        title_to_sents[title] = sents

    # Unique supporting titles (preserve order)
    seen = set()
    supporting_titles = []
    for t in sf["title"]:
        if t not in seen:
            seen.add(t)
            supporting_titles.append(t)

    docs = []
    for title in supporting_titles:
        if title in title_to_sents:
            sents = title_to_sents[title]
            docs.append({
                "title": title,
                "text": " ".join(sents),
                "sentences": sents,
            })
    return docs


def extract_supporting_sentences(example: dict) -> list[dict]:
    """
    Extract *only* the gold supporting sentences (sentence-level Oracle).
    """
    sf = example["supporting_facts"]
    ctx = example["context"]

    title_to_sents: dict[str, list[str]] = {}
    for title, sents in zip(ctx["title"], ctx["sentences"]):
        title_to_sents[title] = sents

    title_to_indices: dict[str, set[int]] = defaultdict(set)
    for title, idx in zip(sf["title"], sf["sent_id"]):
        title_to_indices[title].add(idx)

    results = []
    for title, indices in title_to_indices.items():
        if title in title_to_sents:
            sents = title_to_sents[title]
            selected = [sents[i] for i in sorted(indices) if i < len(sents)]
            results.append({"title": title, "text": " ".join(selected)})
    return results


def get_all_context_docs(example: dict) -> list[dict]:
    """
    Return all 10 context passages (for retrieval simulation).
    """
    ctx = example["context"]
    docs = []
    for title, sents in zip(ctx["title"], ctx["sentences"]):
        docs.append({
            "title": title,
            "text": " ".join(sents),
            "sentences": sents,
        })
    return docs


def get_gold_titles(example: dict) -> set[str]:
    """
    Return the set of gold supporting titles.
    """
    return set(example["supporting_facts"]["title"])
