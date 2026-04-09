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
│   └─ Extract text after "Answer:" ──► EM / Token F1 / Semantic Match / FActScore (decompose)
│
└─ Full raw output as-is ──► Citation Grounding / Hallucination / Faithfulness
```

**Example:**

```
Raw output:  "Answer: Greenwich Village, New York City\nCitations: [Doc 2]"

  → Extracted answer:  "Greenwich Village, New York City"   ← used for EM / F1 / Semantic Match
  → Full raw output:   (unchanged)                          ← used for grounding & hallucination metrics
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
| **LLM Hallucination Check** | Full raw output | Hallucination (YES/NO) | 1 LLM call/sample |
| **Faithfulness Score** | Full raw output | Faithfulness (0–1) | 1 LLM call/sample |

### Subset metrics (first 50 samples)

| Metric | Input | Type | Cost |
|--------|-------|------|------|
| **Atomic FActScore** | Extracted answer | Factual precision | ~6 LLM calls/sample |

FActScore first **rewrites Q+A into a declarative statement** (e.g. Q: "In what year?" A: "1755" → "The university was founded in 1755"), then decomposes it into atomic claims and verifies each against the documents. The rewrite step gives claims full context, avoiding false negatives when verifying short answers. Running on a 50-sample subset keeps cost manageable while providing fine-grained hallucination analysis.

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
│   │   └── __init__.py               # BM25, Contriever, Hybrid (E1–E3)
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
│   └── e_oracle.py                   # E-Oracle: Oracle RAG upper bound
│
└── outputs/                          # Experiment results (gitignored)
    └── e_oracle/
        └── <model_key>/              # Per-model subdirectory
            ├── run_config.json       # Run configuration
            ├── results.jsonl         # Per-example results
            └── metrics.json          # Aggregate metrics
```

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
python -m experiments.e_oracle --dry_run
```

This runs 3 examples end-to-end to verify the pipeline works.

## Running Experiments

All experiments are run from the project root directory:

```bash
# E-Oracle with default model (gpt-4o-mini via OpenAI)
python -m experiments.e_oracle

# E-Oracle via OpenRouter
python -m experiments.e_oracle --model or/gpt-4o-mini

# Custom sample size
python -m experiments.e_oracle --sample_size 100

# Dry run (3 examples only, for testing)
python -m experiments.e_oracle --dry_run

# Skip atomic FActScore (only full-scale metrics)
python -m experiments.e_oracle --factscore_n 0

# Custom FActScore subset size (default: 50)
python -m experiments.e_oracle --factscore_n 100

# Start fresh, ignoring previous results
python -m experiments.e_oracle --no_resume
```

### Output directory layout

Results are saved per-experiment, per-model:

```
outputs/e_oracle/
├── gpt-4o-mini/          # --model gpt-4o-mini
├── or_gpt-4o-mini/       # --model or/gpt-4o-mini
├── or_deepseek-v3/       # --model or/deepseek-v3
└── or_llama-3-8b/        # --model or/llama-3-8b
```

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

## Key Design Decisions

- **Structured output + dual-branch evaluation**: RAG prompts require the LLM to output `Answer: ...` and `Citations: ...` on separate lines. The extracted short answer feeds EM/F1/Semantic Match/FActScore; the full raw output feeds citation grounding and hallucination.
- **Two-tier evaluation**: Lightweight hallucination metrics run on all 500 samples; expensive atomic FActScore runs on a configurable subset (default 50). This balances cost and coverage.
- **Unified LLM interface**: All providers use OpenAI-compatible APIs, so `src/generation.py` is a single thin wrapper.
- **OpenRouter support**: A single `OPENROUTER_API_KEY` can access all models, useful when direct API access is unavailable.
- **Per-model output directories**: Each model's results are isolated under `outputs/<experiment>/<model_key>/`, preventing cross-contamination.
- **Safe resume**: Resume validates config before reusing results. Mismatched configs trigger an error instead of silently deleting data.
- **Fixed seed sampling**: `seed=42` ensures all experiments use the same 500 HotpotQA examples for fair comparison.
- **Lazy model loading**: The sentence-transformers model for Semantic Match is loaded on first use, not at import time.