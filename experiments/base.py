"""
Shared experiment runner — eliminates boilerplate across E0, E-Oracle, E1, etc.

Each experiment script provides a lightweight ``prepare_fn(example) -> dict``
that returns the system/user prompts and evaluation context for a single
HotpotQA example.  Everything else (data loading, resume, generation loop,
metric computation, aggregation, summary printing) is handled here.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Callable

from tqdm import tqdm

# Ensure project root is on sys.path (idempotent)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.data import load_hotpotqa
from src.generation import generate
from src.evaluation import compute_all_metrics
from src.evaluation.metrics import aggregate_metrics, parse_structured_output

logger = logging.getLogger(__name__)


def run_experiment(
    experiment_name: str,
    prepare_fn: Callable[[dict], dict],
    run_cfg: dict,
    sample_size: int = config.HOTPOTQA_SAMPLE_SIZE,
    model_key: str = config.DEFAULT_MODEL,
    output_dir: Path | None = None,
    dry_run: bool = False,
    resume: bool = True,
    factscore_n: int = 0,
) -> dict:
    """
    Generic experiment runner.

    Args:
        experiment_name: display name, e.g. "E0-NoRAG", "E-Oracle", "E1-BM25".
        prepare_fn: callable(example) -> dict with keys:
            - system_prompt (str)
            - user_prompt (str)
            - docs (list[dict] | None): reference docs for evaluation
            - gold_titles (set[str] | None): for retrieval / citation metrics
            - retrieved_titles (list[str] | None): for retrieval precision
            - extra_record (dict): additional fields to store in results.jsonl
        run_cfg: experiment config dict (must contain "experiment", "model",
                 "sample_size", "seed" for resume validation).
        sample_size: number of HotpotQA examples.
        model_key: LLM identifier from config.MODELS.
        output_dir: where to save results.
        dry_run: if True, process only the first 3 examples.
        resume: if True, skip already-completed examples.
        factscore_n: compute atomic FActScore on the first N examples (0 = skip).

    Returns:
        Aggregated metrics dict.
    """

    # ----------------------------------- setup ---------------------------------- #
    if output_dir is None:
        raise ValueError("output_dir must be specified")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "results.jsonl"
    metrics_file = output_dir / "metrics.json"
    run_config_file = output_dir / "run_config.json"

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
                    diffs.append(
                        f"{key}: saved={saved_cfg.get(key)}, current={run_cfg[key]}"
                    )
            logger.error(
                "Config mismatch in %s (%s). "
                "Use --no_resume to start fresh, or use a different config.",
                output_dir,
                "; ".join(diffs),
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
        for example in tqdm(samples, desc=experiment_name):
            if example["id"] in done_ids:
                continue

            # Experiment-specific preparation
            prep = prepare_fn(example)

            # Generate
            try:
                prediction = generate(
                    model_key=model_key,
                    system_prompt=prep["system_prompt"],
                    user_prompt=prep["user_prompt"],
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

            # Decide which LLM-based metrics to compute for this example
            do_factscore = factscore_n > 0 and processed_count < factscore_n

            # Evaluate
            metrics = compute_all_metrics(
                prediction=prediction,
                gold_answer=example["answer"],
                question=example["question"],
                docs=prep.get("docs"),
                gold_titles=prep.get("gold_titles"),
                retrieved_titles=prep.get("retrieved_titles"),
                model_key=model_key,
                compute_hallucination=prep.get("docs") is not None,
                compute_factscore=do_factscore,
            )

            # Record
            parsed = parse_structured_output(prediction)
            record = {
                "id": example["id"],
                "question": example["question"],
                "gold_answer": example["answer"],
                "prediction": prediction,
                "extracted_answer": parsed["answer"],
                "metrics": metrics,
            }
            record.update(prep.get("extra_record", {}))

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            all_results.append(record)
            all_metrics_list.append(metrics)
            processed_count += 1

    # ----------------------------- Aggregate & save ----------------------------- #
    agg = aggregate_metrics(all_metrics_list)
    agg["experiment"] = experiment_name
    agg["model"] = model_key

    # Compute hallucination rate (proportion of examples with hallucination)
    hal_flags = [
        m.get("has_hallucination")
        for m in all_metrics_list
        if m.get("has_hallucination") is not None
    ]
    if hal_flags:
        agg["hallucination_rate"] = sum(hal_flags) / len(hal_flags)

    # Aggregate FActScore separately (subset only)
    fs_scores = [m["factscore"] for m in all_metrics_list if "factscore" in m]
    if fs_scores:
        agg["factscore_n"] = len(fs_scores)
        agg["mean_factscore"] = sum(fs_scores) / len(fs_scores)

    metrics_file.write_text(json.dumps(agg, indent=2, ensure_ascii=False))

    # ------------------------------- Print summary ------------------------------ #
    print("\n" + "=" * 60)
    print(f"{experiment_name} Results Summary")
    print("=" * 60)
    print(f"  Samples evaluated : {agg['n']}")
    print(f"  Exact Match       : {agg.get('mean_em', 0):.4f}")
    print(f"  Token F1          : {agg.get('mean_token_f1', 0):.4f}")
    print(f"  Semantic Match    : {agg.get('mean_semantic_match', 0):.4f}")
    if agg.get("mean_retrieval_precision_at_5") is not None:
        print(f"  Precision@5       : {agg['mean_retrieval_precision_at_5']:.4f}")
    if agg.get("mean_citation_grounding_rate") is not None:
        print(f"  Citation Grounding: {agg['mean_citation_grounding_rate']:.4f}")
    if "hallucination_rate" in agg:
        print(f"  Hallucination Rate: {agg['hallucination_rate']:.4f}")
    if agg.get("mean_faithfulness") is not None:
        print(f"  Faithfulness      : {agg['mean_faithfulness']:.4f}")
    if "mean_factscore" in agg:
        print(f"  FActScore (n={agg['factscore_n']})"
              f"  : {agg['mean_factscore']:.4f}")
    print(f"  Results saved to  : {output_dir}")
    print("=" * 60)

    return agg
