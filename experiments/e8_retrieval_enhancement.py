"""
E8: Retrieval Enhancement
===========================
Compare retrieval enhancement methods against E3 baseline hybrid retrieval.
Supported enhancement modes:
- baseline: E3 original hybrid retrieval (BM25 + Contriever + RRF)
- query_rewrite: Query rewriting before hybrid retrieval
- rerank: Cross-Encoder reranking after hybrid retrieval
- full: Query rewriting + Cross-Encoder reranking
Usage:
    python -m experiments.e8_retrieval_enhancement --mode baseline [--dry_run]
    python -m experiments.e8_retrieval_enhancement --mode query_rewrite [--dry_run]
    python -m experiments.e8_retrieval_enhancement --mode rerank [--dry_run]
    python -m experiments.e8_retrieval_enhancement --mode full [--dry_run]
"""
import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.data import get_all_context_docs, get_gold_titles
from src.retrieval import hybrid_retrieve, bm25_retrieve, dense_retrieve
from src.retrieval.fusion import rrf_fusion
from src.prompts import (
    RAG_SYSTEM_PROMPT, build_rag_user_prompt,
    QUERY_REWRITE_SYSTEM_PROMPT, build_query_rewrite_prompt
)
from src.generation import generate
from experiments.base import run_experiment

logger = logging.getLogger(__name__)

# Global Cross-Encoder model (lazy loading)
_rerank_model = None
def _get_rerank_model():
    global _rerank_model
    if _rerank_model is None:
        from sentence_transformers import CrossEncoder
        _rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _rerank_model

def cross_encoder_rerank(query: str, docs: list[dict], top_k: int = 5) -> list[dict]:
    """Rerank documents with Cross-Encoder and return top-k."""
    if not docs:
        return []
    model = _get_rerank_model()
    # Prepare query-document pairs
    pairs = [[query, f"{d['title']} {d['text']}"] for d in docs]
    # Predict relevance scores
    scores = model.predict(pairs)
    # Add scores to docs and sort
    scored_docs = []
    for idx, doc in enumerate(docs):
        scored_doc = dict(doc)
        scored_doc["rerank_score"] = float(scores[idx])
        scored_docs.append(scored_doc)
    # Sort by rerank score descending, take top-k
    scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored_docs[:top_k]

def _make_prepare_retrieval_enhancement(
    mode: str,
    top_k: int,
    rrf_k: int = 60,
    rewrite_top_k: int = 10,
    model_key: str = config.DEFAULT_MODEL,
):
    """Return prepare_fn for different enhancement modes."""
    def _prepare(example: dict) -> dict:
        all_docs = get_all_context_docs(example)
        gold_titles = get_gold_titles(example)
        original_question = example["question"]
        retrieved_docs = []

        # Mode 1: Baseline (E3 original hybrid retrieval)
        if mode == "baseline":
            retrieved_docs = hybrid_retrieve(
                query=original_question,
                docs=all_docs,
                top_k=top_k,
                rrf_k=rrf_k,
            )

        # Mode 2: Query Rewrite only
        elif mode == "query_rewrite":
            # Step 1: Rewrite query
            rewrite_prompt = build_query_rewrite_prompt(original_question)
            rewritten_queries = generate(
                model_key=model_key,
                system_prompt=QUERY_REWRITE_SYSTEM_PROMPT,
                user_prompt=rewrite_prompt,
            ).splitlines()
            # Filter valid non-empty queries
            rewritten_queries = [q.strip() for q in rewritten_queries if q.strip()]
            # Add original query to ensure coverage
            all_queries = [original_question] + rewritten_queries

            # Step 2: Retrieve for each query and merge results
            all_retrieved = []
            for query in all_queries:
                bm25_all = bm25_retrieve(query, all_docs, top_k=len(all_docs))
                dense_all = dense_retrieve(query, all_docs, top_k=len(all_docs))
                fused = rrf_fusion(bm25_all, dense_all, k=rrf_k)
                all_retrieved.extend(fused[:rewrite_top_k])

            # Step 3: Deduplicate by original_index and rerank by RRF score
            seen_indices = set()
            unique_docs = []
            for doc in all_retrieved:
                if doc["original_index"] not in seen_indices:
                    seen_indices.add(doc["original_index"])
                    unique_docs.append(doc)
            # Sort by max RRF score and take top-k
            unique_docs.sort(key=lambda x: x["rrf_score"], reverse=True)
            retrieved_docs = unique_docs[:top_k]

        # Mode 3: Rerank only
        elif mode == "rerank":
            # Step 1: Get larger candidate pool from hybrid retrieval
            candidate_docs = hybrid_retrieve(
                query=original_question,
                docs=all_docs,
                top_k=rewrite_top_k,  # Get top 10 for reranking
                rrf_k=rrf_k,
            )
            # Step 2: Cross-Encoder rerank to get top-k
            retrieved_docs = cross_encoder_rerank(original_question, candidate_docs, top_k=top_k)

        # Mode 4: Full enhancement (query rewrite + rerank)
        elif mode == "full":
            # Step 1: Query rewrite and retrieve candidate pool
            rewrite_prompt = build_query_rewrite_prompt(original_question)
            rewritten_queries = generate(
                model_key=model_key,
                system_prompt=QUERY_REWRITE_SYSTEM_PROMPT,
                user_prompt=rewrite_prompt,
            ).splitlines()
            rewritten_queries = [q.strip() for q in rewritten_queries if q.strip()]
            all_queries = [original_question] + rewritten_queries

            all_retrieved = []
            for query in all_queries:
                bm25_all = bm25_retrieve(query, all_docs, top_k=len(all_docs))
                dense_all = dense_retrieve(query, all_docs, top_k=len(all_docs))
                fused = rrf_fusion(bm25_all, dense_all, k=rrf_k)
                all_retrieved.extend(fused[:rewrite_top_k])

            # Deduplicate
            seen_indices = set()
            unique_docs = []
            for doc in all_retrieved:
                if doc["original_index"] not in seen_indices:
                    seen_indices.add(doc["original_index"])
                    unique_docs.append(doc)
            unique_docs.sort(key=lambda x: x["rrf_score"], reverse=True)
            candidate_docs = unique_docs[:rewrite_top_k]

            # Step 2: Cross-Encoder rerank
            retrieved_docs = cross_encoder_rerank(original_question, candidate_docs, top_k=top_k)

        else:
            raise ValueError(f"Unsupported mode: {mode}. Choose from baseline/query_rewrite/rerank/full")

        # Return standard format compatible with base run_experiment
        return {
            "system_prompt": RAG_SYSTEM_PROMPT,
            "user_prompt": build_rag_user_prompt(example["question"], retrieved_docs),
            "docs": retrieved_docs,
            "gold_titles": gold_titles,
            "retrieved_titles": [d["title"] for d in retrieved_docs],
            "extra_record": {
                "mode": mode,
                "original_question": original_question,
                "retrieved_docs": [
                    {
                        "title": d["title"],
                        "text": d["text"],
                        "bm25_score": d.get("bm25_score"),
                        "dense_score": d.get("dense_score"),
                        "rrf_score": d.get("rrf_score"),
                        "rerank_score": d.get("rerank_score"),
                        "original_index": d["original_index"],
                    }
                    for d in retrieved_docs
                ],
            },
        }
    return _prepare

def run_e8_retrieval_enhancement(
    mode: str = "baseline",
    sample_size: int = config.HOTPOTQA_SAMPLE_SIZE,
    model_key: str = config.DEFAULT_MODEL,
    output_dir: Path | None = None,
    dry_run: bool = False,
    resume: bool = True,
    factscore_n: int = 50,
    top_k: int = config.TOP_K,
    rrf_k: int = 60,
    rewrite_top_k: int = 10,
):
    """Run E8 retrieval enhancement experiment."""
    valid_modes = ["baseline", "query_rewrite", "rerank", "full"]
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}")

    if output_dir is None:
        model_dir = model_key.replace("/", "_")
        output_dir = config.OUTPUT_DIR / "e8_retrieval_enhancement" / mode / model_dir
    
    run_cfg = {
        "experiment": f"E8-Retrieval-Enhancement-{mode}",
        "model": model_key,
        "mode": mode,
        "sample_size": sample_size,
        "seed": config.RANDOM_SEED,
        "top_k": top_k,
        "rrf_k": rrf_k,
        "rewrite_top_k": rewrite_top_k,
        "description": f"Retrieval enhancement experiment, mode: {mode}",
    }

    return run_experiment(
        experiment_name=f"E8-Retrieval-Enhancement-{mode}",
        prepare_fn=_make_prepare_retrieval_enhancement(mode, top_k, rrf_k, rewrite_top_k, model_key),
        run_cfg=run_cfg,
        sample_size=sample_size,
        model_key=model_key,
        output_dir=output_dir,
        dry_run=dry_run,
        resume=resume,
        factscore_n=factscore_n,
    )

def main():
    parser = argparse.ArgumentParser(description="E8: Retrieval enhancement with query rewriting and reranking")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["baseline", "query_rewrite", "rerank", "full"],
                        help="Enhancement mode to run")
    parser.add_argument("--sample_size", type=int, default=config.HOTPOTQA_SAMPLE_SIZE,
                        help="Number of HotpotQA examples (default 500)")
    parser.add_argument("--model", type=str, default=config.DEFAULT_MODEL,
                        help="LLM model to use (default: gpt-4o-mini)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Run on 3 examples only (for testing)")
    parser.add_argument("--no_resume", action="store_true",
                        help="Start fresh, ignore previous results")
    parser.add_argument("--factscore_n", type=int, default=50,
                        help="Compute atomic FActScore on the first N examples (0 to skip)")
    parser.add_argument("--top_k", type=int, default=config.TOP_K,
                        help="Number of final documents to retrieve (default: 5)")
    parser.add_argument("--rrf_k", type=int, default=60,
                        help="RRF constant (default: 60)")
    parser.add_argument("--rewrite_top_k", type=int, default=10,
                        help="Number of candidate docs for rewrite/rerank (default: 10)")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_e8_retrieval_enhancement(
        mode=args.mode,
        sample_size=args.sample_size,
        model_key=args.model,
        dry_run=args.dry_run,
        resume=not args.no_resume,
        factscore_n=args.factscore_n,
        top_k=args.top_k,
        rrf_k=args.rrf_k,
        rewrite_top_k=args.rewrite_top_k,
    )

if __name__ == "__main__":
    main()