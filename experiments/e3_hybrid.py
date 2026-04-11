"""
E3: Hybrid Retrieval (BM25 + Contriever + RRF)
================================================
For each HotpotQA question, retrieve with BM25 and Contriever, fuse using RRF,
and feed top-5 to LLM.

Usage:
    python -m experiments.e3_hybrid [--sample_size 500] [--model or/gpt-4o-mini] [--dry_run]
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.data import get_all_context_docs, get_gold_titles
from src.retrieval import hybrid_retrieve
from src.prompts import RAG_SYSTEM_PROMPT, build_rag_user_prompt
from experiments.base import run_experiment

logger = logging.getLogger(__name__)


def _make_prepare_hybrid(top_k: int, rrf_k: int = 60):
    """Return a prepare_fn that uses hybrid retrieval with given top_k and rrf_k."""

    def _prepare(example: dict) -> dict:
        all_docs = get_all_context_docs(example)
        gold_titles = get_gold_titles(example)
        retrieved_docs = hybrid_retrieve(
            query=example["question"],
            docs=all_docs,
            top_k=top_k,
            rrf_k=rrf_k,
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
                        "bm25_score": d.get("bm25_score"),
                        "dense_score": d.get("dense_score"),
                        "rrf_score": d["rrf_score"],
                        "original_index": d["original_index"],
                    }
                    for d in retrieved_docs
                ],
            },
        }

    return _prepare


def run_e3_hybrid(
    sample_size: int = config.HOTPOTQA_SAMPLE_SIZE,
    model_key: str = config.DEFAULT_MODEL,
    output_dir: Path | None = None,
    dry_run: bool = False,
    resume: bool = True,
    factscore_n: int = 0,
    top_k: int = config.TOP_K,
    rrf_k: int = 60,
):
    """Run the E3 hybrid retrieval experiment."""
    if output_dir is None:
        model_dir = model_key.replace("/", "_")
        output_dir = config.OUTPUT_DIR / "e3_hybrid" / model_dir

    run_cfg = {
        "experiment": "E3-Hybrid",
        "model": model_key,
        "sample_size": sample_size,
        "seed": config.RANDOM_SEED,
        "top_k": top_k,
        "rrf_k": rrf_k,
        "retriever": "BM25+Contriever+RRF",
        "description": "Hybrid retrieval: BM25 + Contriever fused by RRF",
    }

    return run_experiment(
        experiment_name="E3-Hybrid",
        prepare_fn=_make_prepare_hybrid(top_k, rrf_k),
        run_cfg=run_cfg,
        sample_size=sample_size,
        model_key=model_key,
        output_dir=output_dir,
        dry_run=dry_run,
        resume=resume,
        factscore_n=factscore_n,
    )


def main():
    parser = argparse.ArgumentParser(description="E3: Hybrid retrieval with RRF")
    parser.add_argument("--sample_size", type=int, default=config.HOTPOTQA_SAMPLE_SIZE)
    parser.add_argument("--model", type=str, default=config.DEFAULT_MODEL)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--factscore_n", type=int, default=50)
    parser.add_argument("--top_k", type=int, default=config.TOP_K)
    parser.add_argument("--rrf_k", type=int, default=60,
                        help="RRF constant (default: 60)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_e3_hybrid(
        sample_size=args.sample_size,
        model_key=args.model,
        dry_run=args.dry_run,
        resume=not args.no_resume,
        factscore_n=args.factscore_n,
        top_k=args.top_k,
        rrf_k=args.rrf_k,
    )


if __name__ == "__main__":
    main()