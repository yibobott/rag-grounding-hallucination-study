"""
Centralized configuration for the RAG grounding & hallucination study.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Model registry (all OpenAI-compatible)
MODELS = {
    "gpt-4o-mini": {
        "provider": "openai",
        "model_name": "gpt-4o-mini",
        "api_key": OPENAI_API_KEY,
        "base_url": None,
        "temperature": 0.0,
        "max_tokens": 512,
    },
    "deepseek-v3": {
        "provider": "deepseek",
        "model_name": "deepseek-chat",
        "api_key": DEEPSEEK_API_KEY,
        "base_url": "https://api.deepseek.com",
        "temperature": 0.0,
        "max_tokens": 512,
    },
    "llama-3-8b": {
        "provider": "groq",
        "model_name": "llama3-8b-8192",
        "api_key": GROQ_API_KEY,
        "base_url": "https://api.groq.com/openai/v1",
        "temperature": 0.0,
        "max_tokens": 512,
    },
    # OpenRouter — use any model available on openrouter.ai
    "or/gpt-4o-mini": {
        "provider": "openrouter",
        "model_name": "openai/gpt-4o-mini",
        "api_key": OPENROUTER_API_KEY,
        "base_url": "https://openrouter.ai/api/v1",
        "temperature": 0.0,
        "max_tokens": 512,
    },
    "or/deepseek-v3": {
        "provider": "openrouter",
        "model_name": "deepseek/deepseek-chat",
        "api_key": OPENROUTER_API_KEY,
        "base_url": "https://openrouter.ai/api/v1",
        "temperature": 0.0,
        "max_tokens": 512,
    },
    "or/llama-3-8b": {
        "provider": "openrouter",
        "model_name": "meta-llama/llama-3-8b-instruct",
        "api_key": OPENROUTER_API_KEY,
        "base_url": "https://openrouter.ai/api/v1",
        "temperature": 0.0,
        "max_tokens": 512,
    },
    "or/deepseek-v3-self-rag": {
    "provider": "openrouter",
    "model_name": "deepseek/deepseek-chat",
    "api_key": OPENROUTER_API_KEY,
    "base_url": "https://openrouter.ai/api/v1",
    "temperature": 0.0,
    "max_tokens": 512,
    "self_rag": True,  # 开启Self-RAG机制
    "self_rag_threshold": 0.7,  # 校验阈值
    "self_rag_retrieval": "hybrid"  # 复用混合检索
    },
}

# Experiment defaults
HOTPOTQA_SAMPLE_SIZE = 500
RANDOM_SEED = 42
TOP_K = 5

# Default generation model for E0, E-Oracle, E1–E3
DEFAULT_MODEL = "gpt-4o-mini"
