"""
E1: BM25 Sparse Retrieval Baseline
====================================
For each HotpotQA question, BM25 ranks the 10 distractor-setting context
documents and selects the top-5. These are fed to GPT-4o-mini for generation.

Usage:
    python -m experiments.e1_bm25 [--sample_size 500] [--model gpt-4o-mini] [--dry_run]

Args:
    sample_size: number of HotpotQA examples to use.
    model: which LLM to use (default: gpt-4o-mini).
    dry_run: if True, only process the first 3 examples (for testing).
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.data import get_all_context_docs, get_gold_titles
from src.retrieval import bm25_retrieve
from src.prompts import RAG_SYSTEM_PROMPT, build_rag_user_prompt
from experiments.base import run_experiment

logger = logging.getLogger(__name__)


def _make_prepare_bm25(top_k: int):
    """Return a prepare_fn that uses BM25 retrieval with the given top_k."""

    def _prepare(example: dict) -> dict:
        all_docs = get_all_context_docs(example)
        gold_titles = get_gold_titles(example)
        retrieved_docs = bm25_retrieve(
            query=example["question"],
            docs=all_docs,
            top_k=top_k,
        )
        return {
            "system_prompt": RAG_SYSTEM_PROMPT,
            "user_prompt": build_rag_user_prompt(example["question"], retrieved_docs),
            "docs": retrieved_docs,
            "gold_titles": gold_titles,
            "retrieved_titles": [d["title"] for d in retrieved_docs],
            "extra_record": {
                "retrieved_docs": [
                    {
                        "title": d["title"],
                        "text": d["text"],
                        "bm25_score": d["bm25_score"],
                        "original_index": d["original_index"],
                    }
                    for d in retrieved_docs
                ],
            },
        }

    return _prepare


def run_e1_bm25(
    sample_size: int = config.HOTPOTQA_SAMPLE_SIZE,
    model_key: str = config.DEFAULT_MODEL,
    output_dir: Path | None = None,
    dry_run: bool = False,
    resume: bool = True,
    factscore_n: int = 0,
    top_k: int = config.TOP_K,
):
    """Run the E1 BM25 sparse retrieval experiment."""
    if output_dir is None:
        model_dir = model_key.replace("/", "_")
        output_dir = config.OUTPUT_DIR / "e1_bm25" / model_dir

    run_cfg = {
        "experiment": "E1-BM25",
        "model": model_key,
        "sample_size": sample_size,
        "seed": config.RANDOM_SEED,
        "top_k": top_k,
        "description": "BM25 sparse retrieval baseline — re-rank 10 distractor docs, top-5",
    }

    return run_experiment(
        experiment_name="E1-BM25",
        prepare_fn=_make_prepare_bm25(top_k),
        run_cfg=run_cfg,
        sample_size=sample_size,
        model_key=model_key,
        output_dir=output_dir,
        dry_run=dry_run,
        resume=resume,
        factscore_n=factscore_n,
    )


def main():
    parser = argparse.ArgumentParser(description="E1: BM25 sparse retrieval baseline")
    parser.add_argument("--sample_size", type=int, default=config.HOTPOTQA_SAMPLE_SIZE)
    parser.add_argument("--model", type=str, default=config.DEFAULT_MODEL)
    parser.add_argument("--dry_run", action="store_true",
                        help="Run on 3 examples only (for testing)")
    parser.add_argument("--no_resume", action="store_true",
                        help="Start fresh, ignore previous results")
    parser.add_argument("--factscore_n", type=int, default=50,
                        help="Compute atomic FActScore on the first N examples (default: 50, 0 to skip)")
    parser.add_argument("--top_k", type=int, default=config.TOP_K,
                        help="Number of documents to retrieve (default: 5)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_e1_bm25(
        sample_size=args.sample_size,
        model_key=args.model,
        dry_run=args.dry_run,
        resume=not args.no_resume,
        factscore_n=args.factscore_n,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
