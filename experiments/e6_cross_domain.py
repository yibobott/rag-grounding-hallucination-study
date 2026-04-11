"""
E6: Cross-Domain Evaluation
=============================
Apply the best-performing RAG configuration (E3 hybrid retrieval) from HotpotQA
to the PubMedQA biomedical dataset, to test cross-domain generalization.
Usage:
    python -m experiments.e6_cross_domain [--sample_size 500] [--model gpt-4o-mini] [--dry_run]
Args:
    sample_size: number of PubMedQA examples to use (default 500).
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
from src.data import load_pubmedqa
from src.retrieval import hybrid_retrieve
from src.prompts import RAG_SYSTEM_PROMPT, build_rag_user_prompt
from src.generation import generate
from src.evaluation import compute_all_metrics
from src.evaluation.metrics import aggregate_metrics, parse_structured_output

logger = logging.getLogger(__name__)


def _prepare_pubmedqa_example(example: dict, top_k: int, rrf_k: int = 60) -> dict:
    """
    Prepare a single PubMedQA example for RAG:
    - Convert PubMedQA context into standard doc format
    - Run hybrid retrieval to get top-k relevant docs
    """
    # Convert PubMedQA context into standard doc list (compatible with retrieval)
    context = example["context"]
    all_docs = []
    # PubMedQA context is a list of context passages; format into standard doc structure
    for idx, ctx_text in enumerate(context, start=1):
        all_docs.append({
            "title": f"PubMed Context {idx}",
            "text": ctx_text,
            "sentences": ctx_text.split(". "),
        })
    
    # Gold docs for evaluation: full PubMed context (all passages)
    gold_titles = {d["title"] for d in all_docs}
    
    # Run hybrid retrieval (best config from HotpotQA)
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
            "gold_long_answer": example["long_answer"],
        },
    }


def run_e6_cross_domain(
    sample_size: int = 500,
    model_key: str = config.DEFAULT_MODEL,
    output_dir: Path | None = None,
    dry_run: bool = False,
    resume: bool = True,
    factscore_n: int = 50,
    top_k: int = config.TOP_K,
    rrf_k: int = 60,
    seed: int = config.RANDOM_SEED,
):
    """Run the E6 cross-domain evaluation on PubMedQA."""
    # Setup output directory
    if output_dir is None:
        model_dir = model_key.replace("/", "_")
        output_dir = config.OUTPUT_DIR / "e6_cross_domain_pubmedqa" / model_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "results.jsonl"
    metrics_file = output_dir / "metrics.json"
    run_config_file = output_dir / "run_config.json"

    # Load PubMedQA dataset
    logger.info("Loading PubMedQA labeled set (%d samples)...", sample_size)
    samples = load_pubmedqa(sample_size=sample_size, seed=seed)
    if dry_run:
        samples = samples[:3]
        logger.info("Dry-run mode: using first %d examples.", len(samples))

    # Resume support: load completed examples
    done_ids: set[str] = set()
    existing_results: list[dict] = []
    if resume and run_config_file.exists() and results_file.exists():
        saved_cfg = json.loads(run_config_file.read_text())
        cfg_match = (
            saved_cfg.get("model") == model_key
            and saved_cfg.get("sample_size") == sample_size
            and saved_cfg.get("seed") == seed
        )
        if cfg_match:
            with open(results_file, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    done_ids.add(rec["id"])
                    existing_results.append(rec)
            logger.info("Resuming: %d examples already completed.", len(done_ids))
        else:
            diffs = []
            for key in ("model", "sample_size", "seed"):
                if saved_cfg.get(key) != locals()[key]:
                    diffs.append(f"{key}: saved={saved_cfg.get(key)}, current={locals()[key]}")
            logger.error(
                "Config mismatch in %s (%s). Use --no_resume to start fresh.",
                output_dir, "; ".join(diffs)
            )
            raise SystemExit(1)

    # Clear previous results if not resuming
    if not resume:
        results_file.unlink(missing_ok=True)

    # Save run config
    run_cfg = {
        "experiment": "E6-Cross-Domain-PubMedQA",
        "model": model_key,
        "sample_size": sample_size,
        "seed": seed,
        "top_k": top_k,
        "rrf_k": rrf_k,
        "retriever": "BM25+Contriever+RRF",
        "dataset": "PubMedQA labeled",
        "description": "Cross-domain evaluation: best HotpotQA RAG config on PubMedQA",
    }
    run_config_file.write_text(json.dumps(run_cfg, indent=2, ensure_ascii=False))

    # Run generation and evaluation loop
    all_results = list(existing_results)
    all_metrics_list: list[dict] = [r["metrics"] for r in existing_results]
    processed_count = len(all_results)

    with open(results_file, "a", encoding="utf-8") as fout:
        for example in tqdm(samples, desc="E6-Cross-Domain-PubMedQA"):
            if example["id"] in done_ids:
                continue

            # Prepare example for RAG
            prep = _prepare_pubmedqa_example(example, top_k=top_k, rrf_k=rrf_k)

            # LLM generation
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

            # Decide whether to compute FActScore
            do_factscore = factscore_n > 0 and processed_count < factscore_n

            # Compute all metrics (gold answer is PubMedQA's final_decision: yes/no/maybe)
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

            # Save result record
            parsed = parse_structured_output(prediction)
            record = {
                "id": example["id"],
                "question": example["question"],
                "gold_answer": example["answer"],
                "gold_long_answer": example["long_answer"],
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

    # Aggregate metrics
    agg = aggregate_metrics(all_metrics_list)
    agg["experiment"] = run_cfg["experiment"]
    agg["model"] = model_key
    agg["dataset"] = "PubMedQA labeled"

    # Compute hallucination rate
    hal_flags = [
        m.get("has_hallucination")
        for m in all_metrics_list
        if m.get("has_hallucination") is not None
    ]
    if hal_flags:
        agg["hallucination_rate"] = sum(hal_flags) / len(hal_flags)

    # Aggregate FActScore
    fs_scores = [m["factscore"] for m in all_metrics_list if "factscore" in m]
    if fs_scores:
        agg["factscore_n"] = len(fs_scores)
        agg["mean_factscore"] = sum(fs_scores) / len(fs_scores)

    # Save aggregated metrics
    metrics_file.write_text(json.dumps(agg, indent=2, ensure_ascii=False))

    # Print result summary
    print("\n" + "=" * 60)
    print(f"E6 Cross-Domain (PubMedQA) Results Summary")
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
    parser = argparse.ArgumentParser(description="E6: Cross-domain evaluation on PubMedQA")
    parser.add_argument("--sample_size", type=int, default=500,
                        help="Number of PubMedQA examples to use (default: 500)")
    parser.add_argument("--model", type=str, default=config.DEFAULT_MODEL,
                        help="LLM model to use (default: gpt-4o-mini)")
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
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED,
                        help="Random seed for sampling (default: 42)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_e6_cross_domain(
        sample_size=args.sample_size,
        model_key=args.model,
        dry_run=args.dry_run,
        resume=not args.no_resume,
        factscore_n=args.factscore_n,
        top_k=args.top_k,
        rrf_k=args.rrf_k,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()