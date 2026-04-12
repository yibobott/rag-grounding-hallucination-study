"""
E7: Self-RAG Inspired Variant
===============================
Build on E3 best hybrid retrieval config, add self-critique and regeneration to reduce hallucination.
Pipeline:
1. Hybrid retrieval (BM25 + Contriever + RRF) to get top-k docs
2. Initial answer generation with standard RAG prompt
3. Self-critique: check for hallucination / unsupported claims
4. Regeneration: fix issues if critique detects hallucination
5. Final answer evaluation (consistent with E0-E6 metrics)
Usage:
    python -m experiments.e7_self_rag [--sample_size 500] [--model gpt-4o-mini] [--dry_run]
Args:
    sample_size: number of HotpotQA examples to use (default 500).
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
from src.data import get_all_context_docs, get_gold_titles, load_hotpotqa
from src.retrieval import hybrid_retrieve
from src.prompts import (
    RAG_SYSTEM_PROMPT, build_rag_user_prompt,
    SELF_RAG_CRITIQUE_SYSTEM_PROMPT, build_critique_prompt,
    SELF_RAG_REGEN_SYSTEM_PROMPT, build_regeneration_prompt
)
from src.generation import generate
from src.evaluation import compute_all_metrics
from src.evaluation.metrics import aggregate_metrics, parse_structured_output
import json
from tqdm import tqdm

logger = logging.getLogger(__name__)

def self_rag_pipeline(
    example: dict,
    model_key: str,
    top_k: int,
    rrf_k: int,
    enable_regeneration: bool
) -> dict:
    """
    Execute full Self-RAG pipeline for a single example.
    Returns a dict ready for evaluation and recording.
    """
    # Step 1: Fixed hybrid retrieval (same as E3 best config)
    all_docs = get_all_context_docs(example)
    gold_titles = get_gold_titles(example)
    retrieved_docs = hybrid_retrieve(
        query=example["question"],
        docs=all_docs,
        top_k=top_k,
        rrf_k=rrf_k,
    )

    # Step 2: Initial answer generation
    system_prompt = RAG_SYSTEM_PROMPT
    user_prompt = build_rag_user_prompt(example["question"], retrieved_docs)
    initial_prediction = generate(
        model_key=model_key,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    # Step 3: Self-critique
    critique_prompt = build_critique_prompt(example["question"], initial_prediction, retrieved_docs)
    critique_result = generate(
        model_key=model_key,
        system_prompt=SELF_RAG_CRITIQUE_SYSTEM_PROMPT,
        user_prompt=critique_prompt,
    )

    # Parse critique result
    has_hallucination = "HasHallucination: YES" in critique_result.upper()
    final_prediction = initial_prediction
    regenerated = False

    # Step 4: Regeneration if enabled and hallucination detected
    if enable_regeneration and has_hallucination:
        regen_prompt = build_regeneration_prompt(
            question=example["question"],
            docs=retrieved_docs,
            original_answer=initial_prediction,
            critique=critique_result
        )
        final_prediction = generate(
            model_key=model_key,
            system_prompt=SELF_RAG_REGEN_SYSTEM_PROMPT,
            user_prompt=regen_prompt,
        )
        regenerated = True

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "docs": retrieved_docs,
        "gold_titles": gold_titles,
        "retrieved_titles": [d["title"] for d in retrieved_docs],
        "initial_prediction": initial_prediction,
        "critique_result": critique_result,
        "has_hallucination_in_initial": has_hallucination,
        "regenerated": regenerated,
        "final_prediction": final_prediction,
    }

def run_e7_self_rag(
    sample_size: int = config.HOTPOTQA_SAMPLE_SIZE,
    model_key: str = config.DEFAULT_MODEL,
    output_dir: Path | None = None,
    dry_run: bool = False,
    resume: bool = True,
    factscore_n: int = 50,
    top_k: int = config.TOP_K,
    rrf_k: int = 60,
    enable_regeneration: bool = True,
):
    """Run E7 Self-RAG variant experiment (custom runner, no base.py patch)."""
    if output_dir is None:
        model_dir = model_key.replace("/", "_")
        suffix = "_with_regen" if enable_regeneration else "_critique_only"
        output_dir = config.OUTPUT_DIR / "e7_self_rag" / f"{model_dir}{suffix}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "results.jsonl"
    metrics_file = output_dir / "metrics.json"
    run_config_file = output_dir / "run_config.json"

    run_cfg = {
        "experiment": "E7-Self-RAG",
        "model": model_key,
        "sample_size": sample_size,
        "seed": config.RANDOM_SEED,
        "top_k": top_k,
        "rrf_k": rrf_k,
        "enable_regeneration": enable_regeneration,
        "retriever": "BM25+Contriever+RRF (fixed)",
        "description": "Self-RAG inspired variant with critique and regeneration",
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
        for example in tqdm(samples, desc="E7-Self-RAG"):
            if example["id"] in done_ids:
                continue

            # Run full Self-RAG pipeline
            try:
                pipeline_output = self_rag_pipeline(
                    example=example,
                    model_key=model_key,
                    top_k=top_k,
                    rrf_k=rrf_k,
                    enable_regeneration=enable_regeneration,
                )
                prediction = pipeline_output["final_prediction"]
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                err_str = str(e)
                if "insufficient_quota" in err_str or "invalid_api_key" in err_str:
                    logger.error("Permanent API error — aborting: %s", e)
                    raise SystemExit(1)
                logger.error("Pipeline failed for %s: %s", example["id"], e)
                prediction = "[ERROR]"
                pipeline_output = {
                    "docs": None,
                    "gold_titles": None,
                    "retrieved_titles": None,
                    "initial_prediction": "[ERROR]",
                    "critique_result": "[ERROR]",
                    "has_hallucination_in_initial": None,
                    "regenerated": False,
                    "final_prediction": "[ERROR]",
                }

            # Decide which LLM-based metrics to compute for this example
            do_factscore = factscore_n > 0 and processed_count < factscore_n

            # Evaluate
            metrics = compute_all_metrics(
                prediction=prediction,
                gold_answer=example["answer"],
                question=example["question"],
                docs=pipeline_output.get("docs"),
                gold_titles=pipeline_output.get("gold_titles"),
                retrieved_titles=pipeline_output.get("retrieved_titles"),
                model_key=model_key,
                compute_hallucination=pipeline_output.get("docs") is not None,
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
                "retrieved_docs": [
                    {
                        "title": d["title"],
                        "text": d["text"],
                        "bm25_score": d.get("bm25_score"),
                        "dense_score": d.get("dense_score"),
                        "rrf_score": d["rrf_score"],
                        "original_index": d["original_index"],
                    }
                    for d in pipeline_output["docs"]
                ] if pipeline_output.get("docs") else None,
                "initial_prediction": pipeline_output["initial_prediction"],
                "critique_result": pipeline_output["critique_result"],
                "has_hallucination_in_initial": pipeline_output["has_hallucination_in_initial"],
                "regenerated": pipeline_output["regenerated"],
            }
            
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            all_results.append(record)
            all_metrics_list.append(metrics)
            processed_count += 1

    # ----------------------------- Aggregate & save ----------------------------- #
    agg = aggregate_metrics(all_metrics_list)
    agg["experiment"] = run_cfg["experiment"]
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
    print(f"E7 Self-RAG Results Summary")
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

def main():
    parser = argparse.ArgumentParser(description="E7: Self-RAG inspired variant with critique and regeneration")
    parser.add_argument("--sample_size", type=int, default=config.HOTPOTQA_SAMPLE_SIZE,
                        help="Number of HotpotQA examples (default 500)")
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
    parser.add_argument("--no_regeneration", action="store_true",
                        help="Disable regeneration, only run critique (ablation)")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_e7_self_rag(
        sample_size=args.sample_size,
        model_key=args.model,
        dry_run=args.dry_run,
        resume=not args.no_resume,
        factscore_n=args.factscore_n,
        top_k=args.top_k,
        rrf_k=args.rrf_k,
        enable_regeneration=not args.no_regeneration,
    )

if __name__ == "__main__":
    main()