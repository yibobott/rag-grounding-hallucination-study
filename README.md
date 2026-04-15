# RAG Grounding & Hallucination Study

> **CS6493 Natural Language Processing — Group SIGMA6**
>
> Topic 4: Retrieval-Augmented Generation for Knowledge-Intensive Tasks

This project systematically investigates how retrieval quality, retrieval strategy, and generation behavior affect answer accuracy, evidence grounding, and hallucination in Retrieval-Augmented Generation (RAG) systems. Nine experiments (E0–E8) progressively explore the design space — from no-retrieval baselines and oracle upper bounds, through sparse/dense/hybrid retrieval, to generation-model comparison, cross-domain transfer, self-critique regeneration, and advanced retrieval enhancements.

## Experiment Overview

| ID | Goal | Configuration | Data |
|----|------|---------------|------|
| E0 | No-RAG baseline | GPT-4o-mini, no retrieval | HotpotQA (500) |
| E-Oracle | Oracle upper bound | Gold supporting passages injected | HotpotQA (500) |
| E1 | Sparse retrieval | BM25 top-5 + GPT-4o-mini | HotpotQA (500) |
| E2 | Dense retrieval | Contriever top-5 + GPT-4o-mini | HotpotQA (500) |
| E3 | Hybrid retrieval | BM25 + Contriever + RRF top-5 + GPT-4o-mini | HotpotQA (500) |
| E4–E5 | Generation comparison | LLaMA-3-8B / GPT-4o-mini / DeepSeek-V3 with fixed hybrid retrieval | HotpotQA (100) |
| E6 | Cross-domain evaluation | Best RAG config (hybrid) applied to biomedical domain | PubMedQA (500) |
| E7 | Self-RAG variant | Self-critique + conditional regeneration on top of hybrid retrieval | HotpotQA (500) |
| E8 | Retrieval enhancement | Query rewriting / Cross-Encoder reranking / full pipeline vs. E3 baseline | HotpotQA (500) |

## Generation Format & Evaluation Data Flow

All RAG experiments prompt the LLM to return a **structured output**:

```
Answer: <short factual answer>
Citations: [Doc N] ...
```

The raw LLM output is then split into two branches for evaluation:

```
Raw LLM output
│
├─ parse_structured_output()
│   └─ Extract text after "Answer:" ──► EM / Token F1 / Semantic Match
│                                    ──► + Question → rewrite → decompose → verify (FActScore)
│
└─ Full raw output as-is ──► Citation Grounding / Hallucination / Faithfulness
```

**Example:**

```
Q: "The director of 'Big Stone Gap' is based in what New York city?"
Raw output: "Answer: Greenwich Village, New York City\nCitations: [Doc 2]"

  → Extracted answer: "Greenwich Village, New York City"
      → EM / F1 / Semantic Match (vs gold answer)
      → + Q → rewrite: "The director of Big Stone Gap is based in Greenwich Village, New York City."
           → decompose → verify against docs (FActScore)

  → Full raw output (unchanged)
      → Citation Grounding / Hallucination / Faithfulness
```

**Fallback:** If the model does not follow the `Answer: / Citations:` format, the parser strips `[Doc N]` tags from the raw output and uses the remaining text as the answer.

## Evaluation Metrics

A **two-tier evaluation** design balances coverage and cost.

### Full-scale metrics (all samples)

| Metric | Input | Type | Cost |
|--------|-------|------|------|
| **Exact Match (EM)** | Extracted answer | Answer accuracy | Free (local) |
| **Token F1** | Extracted answer | Answer accuracy | Free (local) |
| **Semantic Match** | Extracted answer | Answer accuracy | Free (local) |
| **Retrieval Precision@5** | Retrieved docs | Retrieval quality | Free (local) |
| **Citation Grounding Rate** | Full raw output | Evidence grounding | Free (local) |
| **Num Citations / Num Grounded** | Full raw output | Evidence grounding | Free (local) |
| **LLM Hallucination Check** | Full raw output | Hallucination (YES/NO) | 1 LLM call/sample |
| **Faithfulness Score** | Full raw output | Faithfulness (0–1) | 1 LLM call/sample |

### Subset metrics (first 50 samples by default)

| Metric | Input | Type | Cost |
|--------|-------|------|------|
| **Atomic FActScore** | Extracted answer + Question | Factual precision | ~8 LLM calls/sample |

FActScore first **rewrites Q+A into a declarative statement** (e.g. Q: "In what year?" A: "1755" → "The university was founded in 1755"), then decomposes it into atomic claims and verifies each against the documents. The rewrite step provides full context, avoiding false negatives when verifying short answers. Running on a configurable subset (default 50) keeps cost manageable while providing fine-grained hallucination analysis.

> **E0 note:** E0 has no retrieval, so Retrieval Precision@5, Citation Grounding Rate, Num Citations, and Num Grounded are recorded as `null`. Hallucination, Faithfulness, and FActScore are evaluated against **gold supporting passages** — measuring whether the model's parametric answer is consistent with the ground-truth evidence. This enables direct comparison with RAG experiments on the same scale.

### Human evaluation

- **Citation Precision / Recall**: correctness and completeness of supporting citations.
- **Inter-annotator agreement**: Cohen's Kappa measuring annotation consistency.

## Project Structure

```
rag-grounding-hallucination-study/
├── config.py                         # Global configuration (API keys, models, params)
├── requirements.txt
├── .env.example                      # API key template
├── .gitignore
├── run_e7_e8.sh                      # Convenience script for running E7 & E8
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py               # Unified re-exports
│   │   ├── hotpotqa.py               # HotpotQA loading + oracle/context extraction
│   │   └── pubmedqa.py               # PubMedQA loading (E6)
│   ├── retrieval/
│   │   ├── __init__.py               # Unified re-exports (bm25_retrieve, dense_retrieve, ...)
│   │   ├── bm25.py                   # BM25 sparse retrieval (E1)
│   │   ├── dense.py                  # Contriever dense retrieval (E2)
│   │   └── fusion.py                 # RRF fusion + hybrid retrieval (E3)
│   ├── evaluation/
│   │   ├── __init__.py               # compute_all_metrics unified entry point
│   │   ├── metrics.py                # Output parser, EM, F1, Semantic Match, Precision@K
│   │   ├── citation.py               # Citation parsing + Grounding Rate
│   │   ├── hallucination.py          # LLM hallucination check + faithfulness score
│   │   └── factscore.py              # Atomic FActScore (LLM-based claim verification)
│   ├── prompts.py                    # Prompt templates (RAG / No-RAG / Self-RAG / Query Rewrite)
│   ├── generation.py                 # Unified LLM interface (OpenAI-compatible)
│   └── self_rag.py                   # Self-RAG generation pipeline (alternative implementation)
│
├── experiments/
│   ├── __init__.py
│   ├── base.py                       # Shared experiment runner (data load, resume, eval, summary)
│   ├── e0_no_rag.py                  # E0: No-RAG baseline
│   ├── e_oracle.py                   # E-Oracle: Oracle RAG upper bound
│   ├── e1_bm25.py                    # E1: BM25 sparse retrieval
│   ├── e2_dense.py                   # E2: Contriever dense retrieval
│   ├── e3_hybrid.py                  # E3: Hybrid retrieval (BM25 + Contriever + RRF)
│   ├── e4_e5_generation_comparison.py # E4–E5: LLM generation comparison
│   ├── e6_cross_domain.py            # E6: Cross-domain evaluation on PubMedQA
│   ├── e7_self_rag.py                # E7: Self-RAG with critique + regeneration
│   └── e8_retrieval_enhancement.py   # E8: Query rewriting + Cross-Encoder reranking
│
├── data/                             # Local data cache (gitignored, auto-generated)
│   └── hotpotqa_500_seed42.json      # Cached 500 HotpotQA samples
│
└── outputs/                          # Experiment results (gitignored)
    ├── e0_no_rag/<model_key>/
    ├── e_oracle/<model_key>/
    ├── e1_bm25/<model_key>/
    ├── e2_dense/<model_key>/
    ├── e3_hybrid/<model_key>/
    ├── e4_e5_generation_comparison/<model_key>/
    ├── e6_cross_domain_pubmedqa/<model_key>/
    ├── e7_self_rag/<model_key>_{with_regen|critique_only}/
    └── e8_retrieval_enhancement/{mode}/<model_key>/
        ├── run_config.json           # Run configuration
        ├── results.jsonl             # Per-example results
        └── metrics.json              # Aggregate metrics
```

## Data

- **HotpotQA** (dev, distractor setting): 500 examples sampled with `seed=42`, fixed across all experiments for fair comparison.
- **PubMedQA** (labeled set, E6 cross-domain): loaded separately via `src/data/pubmedqa.py`.

The 500 HotpotQA samples are drawn from the validation split using `random.sample` with a fixed seed, so every experiment uses the exact same subset. On the first call, `load_hotpotqa()` downloads data from HuggingFace and caches it locally to `data/hotpotqa_500_seed42.json`. Subsequent calls load directly from this local file (fast, no network required).

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

Copy the template and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=sk-your-openai-key
DEEPSEEK_API_KEY=sk-your-deepseek-key
GROQ_API_KEY=gsk_your-groq-key
OPENROUTER_API_KEY=sk-or-v1-your-openrouter-key
```

> - Only `OPENAI_API_KEY` is required for E0–E3 (direct OpenAI access).
> - `DEEPSEEK_API_KEY` and `GROQ_API_KEY` are needed for E4–E5 (direct provider access).
> - `OPENROUTER_API_KEY` is an alternative: a single key to access all models via [OpenRouter](https://openrouter.ai).

### 3. Verify setup (dry run)

```bash
cd rag-grounding-hallucination-study
python -m experiments.e0_no_rag --dry_run
python -m experiments.e_oracle --dry_run
python -m experiments.e1_bm25 --dry_run
```

This runs 3 examples end-to-end to verify the pipeline works.

## Running Experiments

All experiments are run from the project root directory.

### E0 — No-RAG Baseline

```bash
python -m experiments.e0_no_rag                        # Default model (gpt-4o-mini)
python -m experiments.e0_no_rag --model or/gpt-4o-mini # Via OpenRouter
python -m experiments.e0_no_rag --dry_run               # Dry run (3 examples)
python -m experiments.e0_no_rag --sample_size 100       # Custom sample size
python -m experiments.e0_no_rag --factscore_n 0         # Skip FActScore
python -m experiments.e0_no_rag --no_resume             # Start fresh
```

> E0 evaluates the LLM's parametric knowledge without any retrieval. Retrieval and citation metrics are `null`. Hallucination metrics use gold supporting passages as reference.

### E-Oracle — Oracle RAG Upper Bound

```bash
python -m experiments.e_oracle                          # Default model
python -m experiments.e_oracle --model or/gpt-4o-mini   # Via OpenRouter
python -m experiments.e_oracle --dry_run
```

### E1 — BM25 Sparse Retrieval

```bash
python -m experiments.e1_bm25                           # Default model + top-5
python -m experiments.e1_bm25 --model or/gpt-4o-mini --top_k 3
python -m experiments.e1_bm25 --dry_run
```

### E2 — Dense Retrieval (Contriever)

```bash
python -m experiments.e2_dense                          # Default model + top-5
python -m experiments.e2_dense --model or/gpt-4o-mini
python -m experiments.e2_dense --top_k 3
python -m experiments.e2_dense --dry_run
```

### E3 — Hybrid Retrieval (BM25 + Contriever + RRF)

```bash
python -m experiments.e3_hybrid                         # Default model + top-5
python -m experiments.e3_hybrid --model or/gpt-4o-mini
python -m experiments.e3_hybrid --rrf_k 60              # RRF constant (default: 60)
python -m experiments.e3_hybrid --dry_run
```

### E4–E5 — Generation Model Comparison

Fixes the best retrieval configuration (E3 hybrid) and compares different LLMs on a 100-sample subset.

```bash
python -m experiments.e4_e5_generation_comparison --model or/gpt-4o-mini
python -m experiments.e4_e5_generation_comparison --model or/deepseek-v3
python -m experiments.e4_e5_generation_comparison --model or/llama-3-8b
python -m experiments.e4_e5_generation_comparison --sample_size 100 --dry_run
```

### E6 — Cross-Domain Evaluation (PubMedQA)

Applies the hybrid retrieval pipeline to PubMedQA biomedical QA data. Constructs a retrieval pool by combining gold context passages with distractor passages sampled from other PubMedQA examples.

```bash
python -m experiments.e6_cross_domain                              # Default (500 samples)
python -m experiments.e6_cross_domain --model or/gpt-4o-mini
python -m experiments.e6_cross_domain --model or/deepseek-v3
python -m experiments.e6_cross_domain --sample_size 200 --dry_run
```

### E7 — Self-RAG (Critique + Regeneration)

Builds on E3 hybrid retrieval and adds a self-critique loop: (1) generate initial answer, (2) critique for hallucination, (3) regenerate if hallucination detected.

```bash
python -m experiments.e7_self_rag --model or/gpt-4o-mini           # Full pipeline (critique + regen)
python -m experiments.e7_self_rag --model or/gpt-4o-mini --no_regeneration  # Critique only (ablation)
python -m experiments.e7_self_rag --dry_run
```

### E8 — Retrieval Enhancement

Compares four retrieval modes against the E3 baseline:

| Mode | Description |
|------|-------------|
| `baseline` | E3 hybrid retrieval (BM25 + Contriever + RRF) |
| `query_rewrite` | LLM-based query rewriting → multi-query hybrid retrieval |
| `rerank` | Hybrid retrieval → Cross-Encoder reranking |
| `full` | Query rewriting + Cross-Encoder reranking |

```bash
python -m experiments.e8_retrieval_enhancement --mode baseline
python -m experiments.e8_retrieval_enhancement --mode query_rewrite --model or/gpt-4o-mini
python -m experiments.e8_retrieval_enhancement --mode rerank --model or/gpt-4o-mini
python -m experiments.e8_retrieval_enhancement --mode full --model or/gpt-4o-mini
python -m experiments.e8_retrieval_enhancement --mode full --dry_run
```

### Common CLI Arguments

All experiments share these flags:

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | LLM model key (see Models section) | `gpt-4o-mini` |
| `--sample_size` | Number of examples to evaluate | `500` (E4–E5: `100`) |
| `--dry_run` | Process only 3 examples for testing | `False` |
| `--no_resume` | Start fresh, discard previous results | `False` |
| `--factscore_n` | FActScore subset size (0 = skip) | `50` |
| `--top_k` | Number of documents to retrieve | `5` |

### Output Directory Layout

Results are saved per-experiment, per-model:

```
outputs/
├── e0_no_rag/or_gpt-4o-mini/
├── e_oracle/or_gpt-4o-mini/
├── e1_bm25/or_gpt-4o-mini/
├── e2_dense/or_gpt-4o-mini/
├── e3_hybrid/or_gpt-4o-mini/
├── e4_e5_generation_comparison/
│   ├── or_gpt-4o-mini/
│   ├── or_deepseek-v3/
│   └── or_llama-3-8b/
├── e6_cross_domain_pubmedqa/
│   ├── or_gpt-4o-mini/
│   └── or_deepseek-v3/
├── e7_self_rag/
│   ├── or_gpt-4o-mini_with_regen/
│   └── or_gpt-4o-mini_critique_only/
└── e8_retrieval_enhancement/
    ├── baseline/or_gpt-4o-mini/
    ├── query_rewrite/or_gpt-4o-mini/
    ├── rerank/or_gpt-4o-mini/
    └── full/or_gpt-4o-mini/
```

### Understanding the Results

Each experiment run produces three files:

| File | Content |
|------|---------|
| `run_config.json` | Run parameters (model, sample_size, seed, retrieval config) |
| `results.jsonl` | Per-example results (one JSON object per line) |
| `metrics.json` | Aggregate metrics across all examples |

**`results.jsonl`** — each line contains:

```json
{
  "id": "5a8e3ea95542995a26add48d",
  "question": "The director of ... is based in what New York city?",
  "gold_answer": "Greenwich Village, New York City",
  "prediction": "Answer: Greenwich Village, New York City\nCitations: [Doc 2]",
  "extracted_answer": "Greenwich Village, New York City",
  "metrics": {
    "em": 1.0, "token_f1": 1.0, "semantic_match": 1.0,
    "retrieval_precision_at_5": 1.0,
    "citation_grounding_rate": 1.0, "num_citations": 1, "num_grounded": 1,
    "has_hallucination": false, "faithfulness": 1.0,
    "factscore": 1.0, "num_claims": 2, "num_supported_claims": 2
  }
}
```

- **`prediction`**: Raw LLM output (RAG experiments include `Answer:` / `Citations:` format; E0 is plain text).
- **`extracted_answer`**: Parsed short answer used for EM / F1 / Semantic Match.
- **`metrics`**: All computed metrics; `factscore` fields only present for the first N samples; `null` fields indicate metrics not applicable to that experiment.
- E7 records additionally include `initial_prediction`, `critique_result`, `has_hallucination_in_initial`, and `regenerated`.
- E8 records additionally include `mode` and `original_question`.

**`metrics.json`** — aggregate means across all examples, e.g.:

```json
{
  "mean_em": 0.486,
  "mean_token_f1": 0.6942,
  "mean_semantic_match": 0.7747,
  "mean_retrieval_precision_at_5": 1.0,
  "mean_citation_grounding_rate": 0.98,
  "mean_num_citations": 1.46,
  "mean_num_grounded": 1.46,
  "mean_has_hallucination": 0.032,
  "mean_faithfulness": 0.961,
  "n": 500,
  "experiment": "E-Oracle",
  "model": "or/gpt-4o-mini",
  "hallucination_rate": 0.032,
  "mean_factscore": 0.9233,
  "factscore_n": 50
}
```

#### Metric Reference

| Field | Meaning |
|-------|---------|
| `mean_em` | Exact string match after normalization |
| `mean_token_f1` | Token-level overlap between prediction and gold |
| `mean_semantic_match` | Cosine similarity of sentence embeddings |
| `mean_retrieval_precision_at_5` | Fraction of top-5 retrieved docs that are gold |
| `mean_citation_grounding_rate` | Fraction of `[Doc N]` citations pointing to gold docs |
| `mean_num_citations` | Average number of `[Doc N]` tags per answer |
| `mean_num_grounded` | Average number of grounded citations |
| `hallucination_rate` | Fraction of samples where LLM judge detected hallucination |
| `mean_faithfulness` | Average faithfulness score (0–1) from LLM judge |
| `mean_factscore` | Fraction of atomic claims supported by documents |
| `factscore_n` | Number of samples used for FActScore computation |

> **Note:** EM is intentionally strict — a correct answer phrased differently from the gold scores 0. Token F1 and Semantic Match provide more forgiving measurements. Always read the three answer-accuracy metrics together.

### Resume Behavior

- Experiments **auto-resume** by default: completed examples are skipped based on their IDs in `results.jsonl`.
- Resume only works when the saved `run_config.json` matches the current run config (model, sample_size, seed).
- On config mismatch, the run **refuses to start** (no data is deleted). Use `--no_resume` to explicitly start fresh.

## Models

All providers use OpenAI-compatible APIs. Models can be accessed either directly or via [OpenRouter](https://openrouter.ai).

| Key | Model | Provider | API |
|-----|-------|----------|-----|
| `gpt-4o-mini` | GPT-4o-mini | OpenAI | Direct |
| `deepseek-v3` | DeepSeek-V3 | DeepSeek | Direct |
| `llama-3-8b` | LLaMA-3-8B | Groq | Direct |
| `or/gpt-4o-mini` | GPT-4o-mini | OpenRouter | `openai/gpt-4o-mini` |
| `or/deepseek-v3` | DeepSeek-V3 | OpenRouter | `deepseek/deepseek-chat` |
| `or/llama-3-8b` | LLaMA-3-8B | OpenRouter | `meta-llama/llama-3-8b-instruct` |

To add a new model, add an entry to `MODELS` in `config.py`. For OpenRouter models, set `model_name` to the [OpenRouter model ID](https://openrouter.ai/models) and `base_url` to `https://openrouter.ai/api/v1`.

## Key Design Decisions

- **Structured output + dual-branch evaluation**: RAG prompts require the LLM to output `Answer: ...` and `Citations: ...` on separate lines. The extracted short answer feeds EM / F1 / Semantic Match / FActScore; the full raw output feeds citation grounding and hallucination evaluation.
- **Two-tier evaluation**: Lightweight metrics run on all samples; expensive atomic FActScore runs on a configurable subset (default 50). This balances cost and coverage.
- **Unified LLM interface**: All providers use OpenAI-compatible APIs, so `src/generation.py` is a single thin wrapper with retry logic.
- **OpenRouter support**: A single `OPENROUTER_API_KEY` can access all models, useful when direct API access is unavailable.
- **Per-model output directories**: Each model's results are isolated under `outputs/<experiment>/<model_key>/`, preventing cross-contamination.
- **Safe resume**: Resume validates config before reusing results. Mismatched configs trigger an error instead of silently deleting data.
- **Fixed seed sampling**: `seed=42` ensures all experiments use the same 500 HotpotQA examples for fair comparison.
- **Local data caching**: `load_hotpotqa()` caches sampled data to `data/hotpotqa_500_seed42.json` on first call. Eliminates repeated HuggingFace downloads.
- **Lazy model loading**: The sentence-transformers model for Semantic Match and the Cross-Encoder for reranking are loaded on first use, not at import time.
- **Permanent error detection**: API errors like `insufficient_quota` or `invalid_api_key` are detected immediately — the program aborts instead of retrying and producing `[ERROR]` results.
- **Modular retrieval stack**: `src/retrieval/` separates BM25, dense (Contriever), and RRF fusion into independent modules, making it straightforward to add or swap retrieval components.