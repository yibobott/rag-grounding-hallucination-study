"""
E0: No-RAG Baseline
====================
GPT-4o-mini answers HotpotQA questions directly without any retrieved
documents.  This establishes the performance floor — what happens when the
model relies entirely on parametric knowledge.

Metrics computed:
- Answer accuracy: EM, Token F1, Semantic Match
- Hallucination: hallucination_check, faithfulness_score (evaluated against
  gold supporting passages — measures whether the model's parametric answer
  happens to be consistent with the ground-truth evidence)
- FActScore: atomic claim verification against gold passages (subset only)

Metrics NOT applicable (skipped):
- Retrieval Precision@5 (no retrieval performed)
- Citation Grounding Rate (no citations expected in No-RAG output)

Usage:
    python -m experiments.e0_no_rag [--sample_size 500] [--model gpt-4o-mini] [--dry_run]

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
from src.data import extract_oracle_docs
from src.prompts import NO_RAG_SYSTEM_PROMPT, build_no_rag_user_prompt
from experiments.base import run_experiment

logger = logging.getLogger(__name__)


def _prepare_e0(example: dict) -> dict:
    """Prepare a No-RAG example: no docs in prompt, oracle docs for evaluation."""
    oracle_docs = extract_oracle_docs(example)
    return {
        "system_prompt": NO_RAG_SYSTEM_PROMPT,
        "user_prompt": build_no_rag_user_prompt(example["question"]),
        # Oracle docs used only for hallucination/faithfulness evaluation,
        # NOT injected into the prompt.
        "docs": oracle_docs if oracle_docs else None,
        "gold_titles": None,       # N/A — no retrieval
        "retrieved_titles": None,   # N/A — no retrieval
        "extra_record": {},
    }


def run_e0_no_rag(
    sample_size: int = config.HOTPOTQA_SAMPLE_SIZE,
    model_key: str = config.DEFAULT_MODEL,
    output_dir: Path | None = None,
    dry_run: bool = False,
    resume: bool = True,
    factscore_n: int = 0,
):
    """Run the E0 No-RAG baseline experiment."""
    if output_dir is None:
        model_dir = model_key.replace("/", "_")
        output_dir = config.OUTPUT_DIR / "e0_no_rag" / model_dir

    run_cfg = {
        "experiment": "E0-NoRAG",
        "model": model_key,
        "sample_size": sample_size,
        "seed": config.RANDOM_SEED,
        "description": "No-RAG baseline — model answers from parametric knowledge only",
    }

    return run_experiment(
        experiment_name="E0-NoRAG",
        prepare_fn=_prepare_e0,
        run_cfg=run_cfg,
        sample_size=sample_size,
        model_key=model_key,
        output_dir=output_dir,
        dry_run=dry_run,
        resume=resume,
        factscore_n=factscore_n,
    )


def main():
    parser = argparse.ArgumentParser(description="E0: No-RAG baseline")
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

    run_e0_no_rag(
        sample_size=args.sample_size,
        model_key=args.model,
        dry_run=args.dry_run,
        resume=not args.no_resume,
        factscore_n=args.factscore_n,
    )


if __name__ == "__main__":
    main()
