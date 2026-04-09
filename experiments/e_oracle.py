"""
E-Oracle: Oracle RAG Upper Bound
=================================
Gold supporting passages from HotpotQA are injected directly into the prompt.
This establishes the performance ceiling for RAG — what happens when retrieval
is perfect.

Usage:
    python -m experiments.e_oracle [--sample_size 500] [--model gpt-4o-mini] [--dry_run]

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
from src.data import extract_oracle_docs, get_gold_titles
from src.prompts import RAG_SYSTEM_PROMPT, build_rag_user_prompt
from experiments.base import run_experiment

logger = logging.getLogger(__name__)


def _prepare_oracle(example: dict) -> dict:
    """Prepare an Oracle example: gold supporting passages as context."""
    oracle_docs = extract_oracle_docs(example)
    gold_titles = get_gold_titles(example)
    return {
        "system_prompt": RAG_SYSTEM_PROMPT,
        "user_prompt": build_rag_user_prompt(example["question"], oracle_docs),
        "docs": oracle_docs,
        "gold_titles": gold_titles,
        "retrieved_titles": [d["title"] for d in oracle_docs],
        "extra_record": {
            "oracle_docs": [{"title": d["title"], "text": d["text"]}
                            for d in oracle_docs],
        },
    }


def run_e_oracle(
    sample_size: int = config.HOTPOTQA_SAMPLE_SIZE,
    model_key: str = config.DEFAULT_MODEL,
    output_dir: Path | None = None,
    dry_run: bool = False,
    resume: bool = True,
    factscore_n: int = 0,
):
    """Run the E-Oracle experiment."""
    if output_dir is None:
        model_dir = model_key.replace("/", "_")
        output_dir = config.OUTPUT_DIR / "e_oracle" / model_dir

    run_cfg = {
        "experiment": "E-Oracle",
        "model": model_key,
        "sample_size": sample_size,
        "seed": config.RANDOM_SEED,
        "description": "Oracle RAG upper bound — gold supporting passages injected",
    }

    return run_experiment(
        experiment_name="E-Oracle",
        prepare_fn=_prepare_oracle,
        run_cfg=run_cfg,
        sample_size=sample_size,
        model_key=model_key,
        output_dir=output_dir,
        dry_run=dry_run,
        resume=resume,
        factscore_n=factscore_n,
    )


def main():
    parser = argparse.ArgumentParser(description="E-Oracle: Oracle RAG upper bound")
    parser.add_argument("--sample_size", type=int, default=config.HOTPOTQA_SAMPLE_SIZE)
    parser.add_argument("--model", type=str, default=config.DEFAULT_MODEL)
    parser.add_argument("--dry_run", action="store_true",
                        help="Run on 3 examples only (for testing)")
    parser.add_argument("--no_resume", action="store_true",
                        help="Start fresh, ignore previous results")
    parser.add_argument("--factscore_n", type=int, default=50,
                        help="Compute atomic FActScore on the first N examples (default: 50, 0 to skip)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_e_oracle(
        sample_size=args.sample_size,
        model_key=args.model,
        dry_run=args.dry_run,
        resume=not args.no_resume,
        factscore_n=args.factscore_n,
    )


if __name__ == "__main__":
    main()
