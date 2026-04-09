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
import json
import logging
import sys
from pathlib import Path
from tqdm import tqdm

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.data import load_hotpotqa, extract_oracle_docs
from src.prompts import NO_RAG_SYSTEM_PROMPT, build_no_rag_user_prompt
from src.generation import generate
from src.evaluation.metrics import (
    exact_match,
    token_f1,
    semantic_match,
    aggregate_metrics,
)
from src.evaluation.hallucination import hallucination_check, faithfulness_score
from src.evaluation.factscore import factscore

logger = logging.getLogger(__name__)


def run_e0_no_rag(
    sample_size: int = config.HOTPOTQA_SAMPLE_SIZE,
    model_key: str = config.DEFAULT_MODEL,
    output_dir: Path | None = None,
    dry_run: bool = False,
    resume: bool = True,
    factscore_n: int = 0,
):
    """
    Run the E0 No-RAG baseline experiment.

    Args:
        sample_size: number of HotpotQA examples to use.
        model_key: which LLM to use (default: gpt-4o-mini).
        output_dir: where to save results.
        dry_run: if True, only process the first 3 examples (for testing).
        resume: if True, skip examples that already have saved results.
        factscore_n: compute atomic FActScore on the first N examples (0 = skip).
    """

    # ----------------------------------- setup ---------------------------------- #
    if output_dir is None:
        model_dir = model_key.replace("/", "_")
        output_dir = config.OUTPUT_DIR / "e0_no_rag" / model_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "results.jsonl"
    metrics_file = output_dir / "metrics.json"
    run_config_file = output_dir / "run_config.json"

    # Run configuration (used for both saving and resume validation)
    run_cfg = {
        "experiment": "E0-NoRAG",
        "model": model_key,
        "sample_size": sample_size,
        "seed": config.RANDOM_SEED,
        "description": "No-RAG baseline — model answers from parametric knowledge only",
    }

    # --------------------------------- Load data -------------------------------- #
    logger.info("Loading HotpotQA dev set (%d samples)...", sample_size)
    samples = load_hotpotqa(sample_size=sample_size, seed=config.RANDOM_SEED)
    if dry_run:
        samples = samples[:3]
        logger.info("Dry-run mode: using first %d examples.", len(samples))

    # ---------------- Resume support: load already-completed IDs ---------------- #
    done_ids: set[str] = set()
    existing_results: list[dict] = []
    if resume and run_config_file.exists() and results_file.exists():
        saved_cfg = json.loads(run_config_file.read_text())
        cfg_match = (
            saved_cfg.get("model") == run_cfg["model"]
            and saved_cfg.get("sample_size") == run_cfg["sample_size"]
            and saved_cfg.get("seed") == run_cfg["seed"]
        )
        if cfg_match:
            with open(results_file) as f:
                for line in f:
                    rec = json.loads(line)
                    done_ids.add(rec["id"])
                    existing_results.append(rec)
            logger.info("Resuming: %d examples already completed.", len(done_ids))
        else:
            diffs = []
            for key in ("model", "sample_size", "seed"):
                if saved_cfg.get(key) != run_cfg[key]:
                    diffs.append(f"{key}: saved={saved_cfg.get(key)}, current={run_cfg[key]}")
            logger.error(
                "Config mismatch in %s (%s). "
                "Use --no_resume to start fresh, or use a different config.",
                output_dir, "; ".join(diffs),
            )
            raise SystemExit(1)

    # When not resuming, clear any previous results to avoid duplicates
    if not resume:
        results_file.unlink(missing_ok=True)

    # Write current run config (after resume check)
    run_config_file.write_text(json.dumps(run_cfg, indent=2))

    # ------------------------ Run generation + evaluation ----------------------- #
    all_results = list(existing_results)
    all_metrics_list: list[dict] = [r["metrics"] for r in existing_results]
    processed_count = len(all_results)

    with open(results_file, "a") as fout:
        for example in tqdm(samples, desc="E0-NoRAG"):
            if example["id"] in done_ids:
                continue

            # Build prompt (no documents)
            user_prompt = build_no_rag_user_prompt(example["question"])

            # Generate
            try:
                prediction = generate(
                    model_key=model_key,
                    system_prompt=NO_RAG_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                )
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                err_str = str(e)
                if "insufficient_quota" in err_str or "invalid_api_key" in err_str:
                    logger.error("Permanent API error — aborting: %s", e)
                    raise SystemExit(1)
                logger.error("Generation failed for %s: %s", example["id"], e)
                prediction = "[ERROR]"

            # ---- Answer accuracy ---- #
            metrics: dict = {
                "em": exact_match(prediction, example["answer"]),
                "token_f1": token_f1(prediction, example["answer"]),
                "semantic_match": semantic_match(prediction, example["answer"]),
                # N/A for No-RAG (no retrieval, no citations)
                "retrieval_precision_at_5": None,
                "citation_grounding_rate": None,
                "num_citations": None,
                "num_grounded": None,
            }

            # ---- Hallucination & faithfulness (against gold passages) ---- #
            # Even though the model did not see these docs, we evaluate
            # whether its parametric answer is consistent with gold evidence.
            # This enables direct comparison with E-Oracle on the same scale.
            oracle_docs = extract_oracle_docs(example)
            if oracle_docs:
                metrics["has_hallucination"] = hallucination_check(
                    prediction, oracle_docs, model_key,
                    question=example["question"],
                )
                metrics["faithfulness"] = faithfulness_score(
                    prediction, oracle_docs, model_key,
                    question=example["question"],
                )

            # ---- FActScore (subset only) ---- #
            do_factscore = factscore_n > 0 and processed_count < factscore_n
            if do_factscore and oracle_docs:
                fs = factscore(
                    prediction, oracle_docs, model_key,
                    question=example["question"],
                )
                metrics["factscore"] = fs["factscore"]
                metrics["num_claims"] = fs["num_claims"]
                metrics["num_supported_claims"] = fs["num_supported"]

            # Record
            record = {
                "id": example["id"],
                "question": example["question"],
                "gold_answer": example["answer"],
                "prediction": prediction,
                "metrics": metrics,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            all_results.append(record)
            all_metrics_list.append(metrics)
            processed_count += 1

    # ----------------------------- Aggregate & save ----------------------------- #
    agg = aggregate_metrics(all_metrics_list)
    agg["n"] = len(all_metrics_list)
    agg["experiment"] = "E0-NoRAG"
    agg["model"] = model_key

    # Explicitly mark N/A metrics as null in aggregate output
    agg["mean_retrieval_precision_at_5"] = None
    agg["mean_citation_grounding_rate"] = None
    agg["mean_num_citations"] = None
    agg["mean_num_grounded"] = None

    # Hallucination rate (proportion of examples with hallucination)
    hal_flags = [m.get("has_hallucination") for m in all_metrics_list
                 if m.get("has_hallucination") is not None]
    if hal_flags:
        agg["hallucination_rate"] = sum(hal_flags) / len(hal_flags)

    # FActScore (subset only)
    fs_scores = [m["factscore"] for m in all_metrics_list if "factscore" in m]
    if fs_scores:
        agg["factscore_n"] = len(fs_scores)
        agg["mean_factscore"] = sum(fs_scores) / len(fs_scores)

    metrics_file.write_text(json.dumps(agg, indent=2, ensure_ascii=False))

    # ------------------------------- Print summary ------------------------------ #
    print("\n" + "=" * 60)
    print("E0 No-RAG Baseline Results Summary")
    print("=" * 60)
    print(f"  Samples evaluated : {agg['n']}")
    print(f"  Exact Match       : {agg.get('mean_em', 0):.4f}")
    print(f"  Token F1          : {agg.get('mean_token_f1', 0):.4f}")
    print(f"  Semantic Match    : {agg.get('mean_semantic_match', 0):.4f}")
    if "hallucination_rate" in agg:
        print(f"  Hallucination Rate: {agg['hallucination_rate']:.4f}")
    if "mean_faithfulness" in agg:
        print(f"  Faithfulness      : {agg['mean_faithfulness']:.4f}")
    if "mean_factscore" in agg:
        print(f"  FActScore (n={agg['factscore_n']})"
              f"  : {agg['mean_factscore']:.4f}")
    print(f"  Results saved to  : {output_dir}")
    print("=" * 60)

    return agg


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
