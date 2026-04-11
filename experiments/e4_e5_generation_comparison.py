"""
E4-E5: Generation Model Comparison
====================================
Fix the best hybrid retrieval configuration (E3: BM25 + Contriever + RRF),
compare the performance of different LLMs on answer accuracy, grounding, and hallucination.
Supported models: LLaMA-3-8B / GPT-4o-mini / DeepSeek-V3
Usage:
    python -m experiments.e4_e5_generation_comparison [--sample_size 100] [--model gpt-4o-mini] [--dry_run]
Args:
    sample_size: number of HotpotQA examples to use (default 100 for generation comparison).
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
from src.retrieval import hybrid_retrieve
from src.prompts import RAG_SYSTEM_PROMPT, build_rag_user_prompt
from experiments.base import run_experiment

logger = logging.getLogger(__name__)


def _make_prepare_generation_compare(top_k: int, rrf_k: int = 60):
    """Return a prepare_fn that uses fixed hybrid retrieval for generation comparison."""
    def _prepare(example: dict) -> dict:
        all_docs = get_all_context_docs(example)
        gold_titles = get_gold_titles(example)
        # Fixed hybrid retrieval (E3 best config)
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


def run_e4_e5_generation_comparison(
    sample_size: int = 100,
    model_key: str = config.DEFAULT_MODEL,
    output_dir: Path | None = None,
    dry_run: bool = False,
    resume: bool = True,
    factscore_n: int = 50,
    top_k: int = config.TOP_K,
    rrf_k: int = 60,
):
    """Run the E4-E5 generation model comparison experiment."""
    if output_dir is None:
        model_dir = model_key.replace("/", "_")
        output_dir = config.OUTPUT_DIR / "e4_e5_generation_comparison" / model_dir
    
    run_cfg = {
        "experiment": "E4-E5-Generation-Comparison",
        "model": model_key,
        "sample_size": sample_size,
        "seed": config.RANDOM_SEED,
        "top_k": top_k,
        "rrf_k": rrf_k,
        "retriever": "BM25+Contriever+RRF (fixed)",
        "description": "Generation model comparison with fixed hybrid retrieval",
    }

    return run_experiment(
        experiment_name="E4-E5-Generation-Comparison",
        prepare_fn=_make_prepare_generation_compare(top_k, rrf_k),
        run_cfg=run_cfg,
        sample_size=sample_size,
        model_key=model_key,
        output_dir=output_dir,
        dry_run=dry_run,
        resume=resume,
        factscore_n=factscore_n,
    )


def main():
    parser = argparse.ArgumentParser(description="E4-E5: Generation model comparison with fixed hybrid retrieval")
    parser.add_argument("--sample_size", type=int, default=100,
                        help="Number of HotpotQA examples (default 100 for generation comparison)")
    parser.add_argument("--model", type=str, default=config.DEFAULT_MODEL,
                        help="LLM model to use (gpt-4o-mini / deepseek-v3 / llama-3-8b)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Run on 3 examples only (for testing)")
    parser.add_argument("--no_resume", action="store_true",
                        help="Start fresh, ignore previous results")
    parser.add_argument("--factscore_n", type=int, default=50,
                        help="Compute atomic FActScore on the first N examples (0 to skip)")
    parser.add_argument("--top_k", type=int, default=config.TOP_K,
                        help="Number of documents to retrieve (default: 5)")
    parser.add_argument("--rrf_k", type=int, default=60,
                        help="RRF constant (default: 60)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_e4_e5_generation_comparison(
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