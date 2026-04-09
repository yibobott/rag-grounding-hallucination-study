"""
Unified LLM generation interface (all providers are OpenAI-compatible).
"""

import time
import logging
from openai import OpenAI, AuthenticationError, RateLimitError

import config

logger = logging.getLogger(__name__)


def get_client(model_key: str) -> tuple[OpenAI, dict]:
    """
    Return an OpenAI client and model config for the given model key.
    """
    cfg = config.MODELS[model_key]
    client = OpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],        # None → default OpenAI endpoint
    )
    return client, cfg


def generate(
    model_key: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> str:
    """
    Call an LLM and return the assistant response text.

    Handles retries with exponential back-off for transient API errors.
    """
    client, cfg = get_client(model_key)
    temp = temperature if temperature is not None else cfg["temperature"]
    mt = max_tokens if max_tokens is not None else cfg["max_tokens"]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=cfg["model_name"],
                messages=messages,
                temperature=temp,
                max_tokens=mt,
            )
            return resp.choices[0].message.content.strip()
        except (AuthenticationError, RateLimitError) as e:
            # Permanent errors: bad key or quota exhausted — no point retrying
            if isinstance(e, AuthenticationError) or "insufficient_quota" in str(e):
                logger.error("Permanent API error for %s: %s", model_key, e)
                raise
            # Transient rate-limit — retry
            logger.warning("Attempt %d/%d failed for %s: %s",
                           attempt, max_retries, model_key, e)
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)
            else:
                raise
        except Exception as e:
            logger.warning("Attempt %d/%d failed for %s: %s",
                           attempt, max_retries, model_key, e)
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)
            else:
                raise


def generate_batch(
    model_key: str,
    system_prompt: str,
    user_prompts: list[str],
    temperature: float | None = None,
    max_tokens: int | None = None,
    delay: float = 0.0,
) -> list[str]:
    """
    Generate responses for a list of user prompts sequentially.

    Args:
        delay: seconds to sleep between requests (rate-limiting).
    """
    results = []
    for prompt in user_prompts:
        resp = generate(model_key, system_prompt, prompt,
                        temperature=temperature, max_tokens=max_tokens)
        results.append(resp)
        if delay > 0:
            time.sleep(delay)
    return results
