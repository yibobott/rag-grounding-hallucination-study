# RAG Grounding & Hallucination Study

> **CS6493 Natural Language Processing — Group SIGMA6**
>
> Topic 4: Retrieval-Augmented Generation for Knowledge-Intensive Tasks

This project systematically investigates how retrieval quality, retrieval design, and generation behavior affect answer accuracy, evidence grounding, and hallucination in RAG systems.

## Experiment Overview

| ID | Goal | Configuration | Data |
|----|------|---------------|------|
| E0 | No-RAG baseline | GPT-4o-mini, no retrieval | HotpotQA |
| E-Oracle | Oracle upper bound | Gold supporting passages injected | HotpotQA |
| E1 | Sparse retrieval | BM25 top-5 + GPT-4o-mini | HotpotQA |
| E2 | Dense retrieval | Contriever top-5 + GPT-4o-mini | HotpotQA |
| E3 | Hybrid retrieval | BM25 + Contriever (RRF) top-5 + GPT-4o-mini | HotpotQA |
| Obs. | Retrieval→Hallucination link | Group E3 results by Precision@5 bins | Re-analysis |
| E4–E5 | Generation comparison | LLaMA-3-8B / GPT-4o-mini / DeepSeek-V3 | HotpotQA (100) |
| E6 | Cross-domain evaluation | Best config on HotpotQA + PubMedQA | HotpotQA + PubMedQA |
| E7 | Self-RAG variant (optional) | Self-RAG-inspired filtering + critique | HotpotQA |
| E8 | Retrieval enhancement | Query rewriting + reranking vs. E3 | HotpotQA |

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

We use a **two-tier evaluation** design to balance coverage and cost.

### Full-scale metrics (all 500 samples)

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

### Subset metrics (first 50 samples)

| Metric | Input | Type | Cost |
|--------|-------|------|------|
| **Atomic FActScore** | Extracted answer + Question | Factual precision | ~8 LLM calls/sample |

FActScore first **rewrites Q+A into a declarative statement** (e.g. Q: "In what year?" A: "1755" → "The university was founded in 1755"), then decomposes it into atomic claims and verifies each against the documents. The rewrite step gives claims full context, avoiding false negatives when verifying short answers. Running on a 50-sample subset keeps cost manageable while providing fine-grained hallucination analysis.

> **E0 note:** E0 has no retrieval, so Retrieval Precision@5, Citation Grounding Rate, Num Citations, and Num Grounded are recorded as `null`. Hallucination, Faithfulness, and FActScore are evaluated against **gold supporting passages** — measuring whether the model's parametric answer happens to be consistent with the ground-truth evidence. This enables direct comparison with E-Oracle and E1–E3 on the same scale.

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
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py               # Unified re-exports
│   │   ├── hotpotqa.py               # HotpotQA loading + oracle/context extraction
│   │   └── pubmedqa.py               # PubMedQA loading (E6)
│   ├── retrieval/
│   │   ├── __init__.py               # Unified re-exports (bm25_retrieve, ...)
│   │   └── bm25.py                  # BM25 sparse retrieval (E1)
│   ├── evaluation/
│   │   ├── __init__.py               # compute_all_metrics unified entry point
│   │   ├── metrics.py                # Output parser, EM, F1, Semantic Match, Precision@K
│   │   ├── citation.py               # Citation parsing + Grounding Rate
│   │   ├── hallucination.py          # LLM hallucination check + faithfulness score
│   │   └── factscore.py              # Atomic FActScore (LLM-based claim verification)
│   ├── prompts.py                    # Prompt templates (RAG / No-RAG / Self-RAG)
│   └── generation.py                 # Unified LLM interface (OpenAI-compatible)
│
├── experiments/
│   ├── __init__.py
│   ├── base.py                       # Shared experiment runner (data load, resume, eval, summary)
│   ├── e0_no_rag.py                  # E0: No-RAG baseline
│   ├── e_oracle.py                   # E-Oracle: Oracle RAG upper bound
│   └── e1_bm25.py                   # E1: BM25 sparse retrieval baseline
│
├── data/                             # Local data cache (gitignored, auto-generated)
│   └── hotpotqa_500_seed42.json     # Cached 500 HotpotQA samples
│
└── outputs/                          # Experiment results (gitignored)
    ├── e0_no_rag/
    │   └── <model_key>/
    ├── e_oracle/
    │   └── <model_key>/
    └── e1_bm25/
        └── <model_key>/
            ├── run_config.json       # Run configuration
            ├── results.jsonl         # Per-example results
            └── metrics.json          # Aggregate metrics
```

## Data

- **HotpotQA** (dev, distractor setting): 500 examples sampled with `seed=42`, fixed across all experiments for fair comparison.
- **PubMedQA** (E6, cross-domain): loaded separately via `src/data/pubmedqa.py`.

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

> - Only `OPENAI_API_KEY` is required for E0, E-Oracle, and E1–E3 (direct OpenAI access).
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

### E0 — No-RAG baseline

```bash
python -m experiments.e0_no_rag                        # Default model (gpt-4o-mini via OpenAI)
python -m experiments.e0_no_rag --model or/gpt-4o-mini # Via OpenRouter
python -m experiments.e0_no_rag --dry_run               # Dry run (3 examples only)
python -m experiments.e0_no_rag --sample_size 100       # Custom sample size
python -m experiments.e0_no_rag --factscore_n 0         # Skip FActScore
python -m experiments.e0_no_rag --factscore_n 100       # Custom FActScore subset
python -m experiments.e0_no_rag --no_resume             # Start fresh
```

> E0 computes answer accuracy (EM, Token F1, Semantic Match) and hallucination metrics (hallucination check, faithfulness, FActScore). Hallucination metrics are evaluated against **gold supporting passages** — measuring whether the model's parametric answer is consistent with the ground-truth evidence. This enables direct comparison with E-Oracle. Retrieval Precision@5 and Citation Grounding Rate are not applicable (no retrieval/citations).

### E-Oracle — Oracle RAG upper bound

```bash
python -m experiments.e_oracle                        # Default model (gpt-4o-mini via OpenAI)
python -m experiments.e_oracle --model or/gpt-4o-mini # Via OpenRouter
python -m experiments.e_oracle --dry_run               # Dry run (3 examples only)
python -m experiments.e_oracle --sample_size 100       # Custom sample size
python -m experiments.e_oracle --factscore_n 0         # Skip FActScore
python -m experiments.e_oracle --factscore_n 100       # Custom FActScore subset
python -m experiments.e_oracle --no_resume             # Start fresh
```

### E1 — BM25 sparse retrieval baseline

```bash
python -m experiments.e1_bm25                                        # Default model + top-5
python -m experiments.e1_bm25 --model or/gpt-4o-mini                 # Via OpenRouter
python -m experiments.e1_bm25 --model or/gpt-4o-mini --top_k 3      # Custom top-k
python -m experiments.e1_bm25 --dry_run                              # Dry run (3 examples only)
python -m experiments.e1_bm25 --model or/gpt-4o-mini --factscore_n 0 # Skip FActScore
python -m experiments.e1_bm25 --model or/gpt-4o-mini --no_resume     # Start fresh
```

### Output directory layout

Results are saved per-experiment, per-model:

```
outputs/
├── e0_no_rag/
│   ├── gpt-4o-mini/          # --model gpt-4o-mini
│   └── or_gpt-4o-mini/       # --model or/gpt-4o-mini
├── e_oracle/
│   ├── gpt-4o-mini/          # --model gpt-4o-mini
│   └── or_gpt-4o-mini/       # --model or/gpt-4o-mini
└── e1_bm25/
    └── or_gpt-4o-mini/       # --model or/gpt-4o-mini
```

### Understanding the results

Each experiment run produces three files under `outputs/<experiment>/<model_key>/`:

| File | Content |
|------|---------|
| `run_config.json` | Run parameters (model, sample_size, seed) |
| `results.jsonl` | Per-example results (one JSON object per line) |
| `metrics.json` | Aggregate metrics across all examples |

**`results.jsonl`** — each line contains:

**E-Oracle / E1–E3 example** (RAG — all metrics available):

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

**E0 example** (No-RAG — retrieval/citation fields are `null`):

```json
{
  "id": "5a8e3ea95542995a26add48d",
  "question": "The director of ... is based in what New York city?",
  "gold_answer": "Greenwich Village, New York City",
  "prediction": "Greenwich Village, New York City",
  "metrics": {
    "em": 1.0, "token_f1": 1.0, "semantic_match": 1.0,
    "retrieval_precision_at_5": null,
    "citation_grounding_rate": null, "num_citations": null, "num_grounded": null,
    "has_hallucination": false, "faithfulness": 1.0,
    "factscore": 1.0, "num_claims": 2, "num_supported_claims": 2
  }
}
```

- **`prediction`**: raw LLM output (RAG experiments include `Answer:` / `Citations:` format; E0 is plain text)
- **`extracted_answer`**: parsed short answer used for EM / F1 / Semantic Match (RAG only; E0 uses raw prediction directly)
- **`metrics`**: all computed metrics for this example; `factscore` fields only present for the first N samples; `null` fields indicate metrics not applicable to that experiment

**`metrics.json`** — aggregate means:

**E-Oracle example:**

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
  "mean_factscore": 0.9233333333333333,
  "mean_num_claims": 1.88,
  "mean_num_supported_claims": 1.76,
  "n": 500,
  "experiment": "E-Oracle",
  "model": "or/gpt-4o-mini",
  "hallucination_rate": 0.032,
  "mean_factscore": 0.9233,
  "factscore_n": 50
}
```

**E0 example** (`null` for retrieval/citation metrics):

```json
{
  "mean_em": 0.21,
  "mean_token_f1": 0.32,
  "mean_semantic_match": 0.45,
  "mean_retrieval_precision_at_5": null,
  "mean_citation_grounding_rate": null,
  "mean_num_citations": null,
  "mean_num_grounded": null,
  "mean_has_hallucination": 0.35,
  "mean_faithfulness": 0.58,
  "n": 500,
  "experiment": "E0-NoRAG",
  "model": "or/gpt-4o-mini",
  "hallucination_rate": 0.35,
  "mean_factscore": 0.62,
  "factscore_n": 50
}
```

Field reference:

| Field | Meaning | E0 | Oracle | E1-BM25 |
|-------|---------|-----|--------|----------|
| `mean_em` | Exact string match after normalization | 0.15–0.30 | 0.4–0.6 | 0.1–0.3 |
| `mean_token_f1` | Overlapping tokens between prediction and gold | 0.25–0.40 | 0.6–0.8 | 0.3–0.5 |
| `mean_semantic_match` | Cosine similarity of sentence embeddings [-1,1] | 0.35–0.55 | 0.7–0.9 | 0.3–0.6 |
| `mean_retrieval_precision_at_5` | Fraction of top-5 retrieved docs that are gold | `null` | 1.0 | 0.3–0.5 |
| `mean_citation_grounding_rate` | Fraction of `[Doc N]` citations pointing to gold documents | `null` | ~1.0 | varies |
| `mean_num_citations` | Average number of `[Doc N]` tags per answer | `null` | — | — |
| `mean_num_grounded` | Average number of citations that point to gold documents | `null` | — | — |
| `mean_has_hallucination` | Fraction of samples where LLM judge detected hallucination | 0.2–0.5 | <0.05 | 0.05–0.15 |
| `hallucination_rate` | Same as `mean_has_hallucination` (convenience alias) | 0.2–0.5 | <0.05 | 0.05–0.15 |
| `mean_faithfulness` | Average faithfulness score (0–1) from LLM judge | 0.4–0.7 | >0.95 | 0.6–0.85 |
| `mean_factscore` | Fraction of atomic claims supported by documents | 0.4–0.7 | >0.9 | 0.6–0.85 |
| `mean_num_claims` | Average number of atomic claims per answer (FActScore subset) | — | — | — |
| `mean_num_supported_claims` | Average number of supported claims (FActScore subset) | — | — | — |
| `n` | Total number of examples evaluated | 500 | 500 | 500 |
| `experiment` | Experiment name | `E0-NoRAG` | `E-Oracle` | `E1-BM25` |
| `model` | Model key used for generation | — | — | — |
| `factscore_n` | Number of samples used for FActScore computation | 50 | 50 | 50 |

> **Note:** EM is intentionally strict — a correct answer phrased differently from the gold (e.g. "Yes, both are opera composers" vs "yes") scores 0. Token F1 and Semantic Match provide more forgiving measurements. Always read the three answer-accuracy metrics together.

### Resume behavior

- Experiments **auto-resume** by default: completed examples are skipped based on their IDs in `results.jsonl`.
- Resume only works when the saved `run_config.json` matches the current run config (model, sample_size, seed).
- On config mismatch, the run **refuses to start** (no data is deleted). Use `--no_resume` to explicitly start fresh.

## Models

All providers use OpenAI-compatible APIs. Models can be accessed either directly or via OpenRouter.

| Key | Model | Provider | API |
|-----|-------|----------|-----|
| `gpt-4o-mini` | GPT-4o-mini | OpenAI | Direct |
| `deepseek-v3` | DeepSeek-V3 | DeepSeek | Direct |
| `llama-3-8b` | LLaMA-3-8B | Groq | Direct |
| `or/gpt-4o-mini` | GPT-4o-mini | OpenRouter | `openai/gpt-4o-mini` |
| `or/deepseek-v3` | DeepSeek-V3 | OpenRouter | `deepseek/deepseek-chat` |
| `or/llama-3-8b` | LLaMA-3-8B | OpenRouter | `meta-llama/llama-3-8b` |

To add a new OpenRouter model, add an entry to `MODELS` in `config.py` with `model_name` set to the [OpenRouter model ID](https://openrouter.ai/models).

## Results & Analysis

All experiments use GPT-4o-mini (`or/gpt-4o-mini` via OpenRouter) on the same 500 HotpotQA samples (`seed=42`).

### Metric comparison (E0 / E1 / E-Oracle)

| Metric | E0 (No-RAG) | E1 (BM25) | E-Oracle | Trend |
|--------|:-----------:|:---------:|:--------:|-------|
| **EM** | 0.132 | 0.322 | 0.486 | E0 < E1 < Oracle |
| **Token F1** | 0.283 | 0.495 | 0.694 | E0 < E1 < Oracle |
| **Semantic Match** | 0.465 | 0.588 | 0.775 | E0 < E1 < Oracle |
| **Retrieval P@5** | N/A | 0.310 | 1.000 | — |
| **Citation Grounding** | N/A | 0.723 | 0.980 | — |
| **Hallucination Rate** | 0.440 | 0.082 | 0.032 | E0 > E1 > Oracle |
| **Faithfulness** | 0.543 | 0.770 | 0.961 | E0 < E1 < Oracle |
| **FActScore** (n=50) | 0.598 | 0.773 | 0.923 | E0 < E1 < Oracle |

### Key findings

**1. Retrieval dramatically reduces hallucination.**
The hallucination rate drops from 44.0% (E0, no retrieval) to 8.2% (E1, BM25 retrieval) — an **81% reduction** — even though BM25 retrieval precision is only 31%. With perfect retrieval (E-Oracle), hallucination falls to 3.2% — a **93% reduction** from E0. This is the central finding: even imperfect retrieval provides strong grounding.

**2. Answer accuracy scales with retrieval quality.**
EM improves 2.4x from E0 (0.132) to E1 (0.322), and 3.7x to E-Oracle (0.486). Token F1 and Semantic Match follow the same monotonic trend. The gap between E1 and E-Oracle (EM +0.164) quantifies the room for improvement through better retrieval methods (E2 dense, E3 hybrid).

**3. Faithfulness and FActScore form a consistent picture.**
Faithfulness (LLM judge, 0–1) and FActScore (atomic claim verification) both rank E0 < E1 < E-Oracle with similar relative gaps. This cross-validation strengthens confidence in the hallucination findings.

**4. The performance spectrum establishes clear baselines.**

```
E0 (floor)           E1 (BM25)             E-Oracle (ceiling)
  Parametric only      Imperfect retrieval    Perfect retrieval
  EM 0.132             EM 0.322              EM 0.486
  Hal. 44.0%           Hal. 8.2%             Hal. 3.2%
  ◄─── retrieval gap ──────────────────────► 
```

E0 and E-Oracle bracket the achievable range. All subsequent retrieval experiments (E2 dense, E3 hybrid, E8 enhanced) are expected to fall between E1 and E-Oracle, with the goal of closing the gap toward the Oracle ceiling.

## Key Design Decisions

- **Structured output + dual-branch evaluation**: RAG prompts require the LLM to output `Answer: ...` and `Citations: ...` on separate lines. The extracted short answer feeds EM/F1/Semantic Match/FActScore; the full raw output feeds citation grounding and hallucination.
- **Two-tier evaluation**: Lightweight hallucination metrics run on all 500 samples; expensive atomic FActScore runs on a configurable subset (default 50). This balances cost and coverage.
- **Unified LLM interface**: All providers use OpenAI-compatible APIs, so `src/generation.py` is a single thin wrapper.
- **OpenRouter support**: A single `OPENROUTER_API_KEY` can access all models, useful when direct API access is unavailable.
- **Per-model output directories**: Each model's results are isolated under `outputs/<experiment>/<model_key>/`, preventing cross-contamination.
- **Safe resume**: Resume validates config before reusing results. Mismatched configs trigger an error instead of silently deleting data.
- **Fixed seed sampling**: `seed=42` ensures all experiments use the same 500 HotpotQA examples for fair comparison.
- **Local data caching**: `load_hotpotqa()` caches sampled data to `data/hotpotqa_500_seed42.json` on first call. Eliminates repeated HuggingFace network requests and speeds up experiment startup.
- **Lazy model loading**: The sentence-transformers model for Semantic Match is loaded on first use, not at import time.
- **Permanent error detection**: API errors like `insufficient_quota` or `invalid_api_key` are detected immediately — the program aborts instead of retrying and producing `[ERROR]` results.