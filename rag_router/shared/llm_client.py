"""Gemini call wrapper: retry/backoff + cost/token tracking, for rag_router.

Standalone copy -- see metrics.py's docstring for why this isn't imported
from ../rag_chatbot/shared/llm_client.py instead. Same lazy-client pattern
as that file (client is built on first real use, not at import time, so
importing this module for testing never requires a live API key).
"""

import logging
import os
import re
import time

logger = logging.getLogger(__name__)

_client = None


def get_client(api_key: str = None):
    from google import genai

    global _client
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "No Gemini API key available. Set GEMINI_API_KEY in the environment, "
            "or pass api_key explicitly."
        )
    if _client is None:
        _client = genai.Client(api_key=key)
    return _client


def build_generation_config(temperature: float, max_output_tokens: int, json_mode: bool = False):
    from google.genai import types

    kwargs = {"temperature": temperature, "max_output_tokens": max_output_tokens}
    if json_mode:
        kwargs["response_mime_type"] = "application/json"
    return types.GenerateContentConfig(**kwargs)


def strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text.removeprefix("json").strip()
    return text


def call_gemini(client, model_name, gen_config, prompt, max_attempts=3, backoff_seconds=2):
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return client.models.generate_content(model=model_name, contents=prompt, config=gen_config)
        except Exception as e:
            last_error = e
            wait = backoff_seconds * attempt
            match = re.search(r"retry in ([\d.]+)s", str(e))
            if match:
                wait = float(match.group(1)) + 1
            logger.warning(f"Gemini call failed (attempt {attempt}/{max_attempts}): {e}")
            if attempt < max_attempts:
                time.sleep(wait)
    raise RuntimeError(f"Gemini call failed after {max_attempts} attempts") from last_error


def extract_token_counts(response):
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return 0, 0
    return usage.prompt_token_count, usage.candidates_token_count


def compute_cost_usd(prompt_tokens, response_tokens, input_rate_per_million, output_rate_per_million):
    return (prompt_tokens / 1_000_000) * input_rate_per_million + (
        response_tokens / 1_000_000
    ) * output_rate_per_million


def log_usage(prompt_tokens, response_tokens, model_name, input_rate_per_million=0.0, output_rate_per_million=0.0):
    import mlflow

    cost_usd = compute_cost_usd(prompt_tokens, response_tokens, input_rate_per_million, output_rate_per_million)
    try:
        with mlflow.start_run(nested=True):
            mlflow.log_param("model", model_name)
            mlflow.log_metric("prompt_tokens", prompt_tokens)
            mlflow.log_metric("response_tokens", response_tokens)
            mlflow.log_metric("total_tokens", prompt_tokens + response_tokens)
            mlflow.log_metric("cost_usd", cost_usd)
    except Exception as e:
        logger.warning(f"MLflow logging failed, continuing without it: {e}")
    return cost_usd
