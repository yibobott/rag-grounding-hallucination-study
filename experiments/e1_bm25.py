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
import json
import logging
import sys
from pathlib import Path
from tqdm import tqdm

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.data import load_hotpotqa, get_all_context_docs, get_gold_titles
from src.retrieval import bm25_retrieve
from src.prompts import RAG_SYSTEM_PROMPT, build_rag_user_prompt
from src.generation import generate
from src.evaluation import compute_all_metrics
from src.evaluation.metrics import aggregate_metrics, parse_structured_output

logger = logging.getLogger(__name__)


def run_e1_bm25(
    sample_size: int = config.HOTPOTQA_SAMPLE_SIZE,
    model_key: str = config.DEFAULT_MODEL,
    output_dir: Path | None = None,
    dry_run: bool = False,
    resume: bool = True,
    factscore_n: int = 0,
    top_k: int = config.TOP_K,
):
    """
    Run the E1 BM25 sparse retrieval experiment.

    Args:
        sample_size: number of HotpotQA examples to use.
        model_key: which LLM to use (default: gpt-4o-mini).
        output_dir: where to save results.
        dry_run: if True, only process the first 3 examples (for testing).
        resume: if True, skip examples that already have saved results.
        factscore_n: compute atomic FActScore on the first N examples (0 = skip).
        top_k: number of documents to retrieve (default: 5).
    """

    # ----------------------------------- setup ---------------------------------- #
    if output_dir is None:
        model_dir = model_key.replace("/", "_")
        output_dir = config.OUTPUT_DIR / "e1_bm25" / model_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "results.jsonl"
    metrics_file = output_dir / "metrics.json"
    run_config_file = output_dir / "run_config.json"

    # Run configuration (used for both saving and resume validation)
    run_cfg = {
        "experiment": "E1-BM25",
        "model": model_key,
        "sample_size": sample_size,
        "seed": config.RANDOM_SEED,
        "top_k": top_k,
        "description": "BM25 sparse retrieval baseline — re-rank 10 distractor docs, top-5",
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
        for example in tqdm(samples, desc="E1-BM25"):
            if example["id"] in done_ids:
                continue

            # Get all 10 context documents and gold titles
            all_docs = get_all_context_docs(example)
            gold_titles = get_gold_titles(example)

            # BM25 retrieval: rank 10 docs, take top-k
            retrieved_docs = bm25_retrieve(
                query=example["question"],
                docs=all_docs,
                top_k=top_k,
            )

            # Build prompt
            user_prompt = build_rag_user_prompt(example["question"], retrieved_docs)

            # Generate
            try:
                prediction = generate(
                    model_key=model_key,
                    system_prompt=RAG_SYSTEM_PROMPT,
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

            # Decide which LLM-based metrics to compute for this example
            do_factscore = factscore_n > 0 and processed_count < factscore_n

            # Evaluate
            retrieved_titles = [d["title"] for d in retrieved_docs]
            metrics = compute_all_metrics(
                prediction=prediction,
                gold_answer=example["answer"],
                question=example["question"],
                docs=retrieved_docs,
                gold_titles=gold_titles,
                retrieved_titles=retrieved_titles,
                model_key=model_key,
                compute_hallucination=True,
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
                "retrieved_docs": [
                    {
                        "title": d["title"],
                        "text": d["text"],
                        "bm25_score": d["bm25_score"],
                        "original_index": d["original_index"],
                    }
                    for d in retrieved_docs
                ],
                "metrics": metrics,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            all_results.append(record)
            all_metrics_list.append(metrics)
            processed_count += 1

    # ----------------------------- Aggregate & save ----------------------------- #
    agg = aggregate_metrics(all_metrics_list)
    agg["experiment"] = "E1-BM25"
    agg["model"] = model_key

    # Compute hallucination rate (proportion of examples with hallucination)
    hal_flags = [m.get("has_hallucination") for m in all_metrics_list
                 if m.get("has_hallucination") is not None]
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
    print("E1-BM25 Results Summary")
    print("=" * 60)
    print(f"  Samples evaluated : {agg['n']}")
    print(f"  Exact Match       : {agg['mean_em']:.4f}")
    print(f"  Token F1          : {agg['mean_token_f1']:.4f}")
    print(f"  Semantic Match    : {agg['mean_semantic_match']:.4f}")
    if "mean_retrieval_precision_at_5" in agg:
        print(f"  Precision@5       : {agg['mean_retrieval_precision_at_5']:.4f}")
    if "mean_citation_grounding_rate" in agg:
        print(f"  Citation Grounding: {agg['mean_citation_grounding_rate']:.4f}")
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
