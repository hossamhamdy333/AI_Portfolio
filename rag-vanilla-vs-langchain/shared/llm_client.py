"""Gemini call wrapper: retry/backoff + cost/token tracking.

This used to be copy-pasted near-verbatim in impl_vanilla/src/generation.py
and impl_vanilla/src/qa_generation.py (same call_gemini, same
compute_cost_usd, same extract_token_counts, same log_usage -- the only
real difference was the prompt template each one built around it). One
copy now; generation.py and qa_generation.py both import this and only
keep their own prompt template + response parsing.
"""

import logging
import os
import re
import time

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

_client = None


def get_client(api_key: str = None) -> "genai.Client":
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
    kwargs = {"temperature": temperature, "max_output_tokens": max_output_tokens}
    if json_mode:
        kwargs["response_mime_type"] = "application/json"
    return types.GenerateContentConfig(**kwargs)


def strip_markdown_fences(text: str) -> str:
    """Gemini in JSON mode still sometimes wraps output in ```json fences --
    strip them before json.loads() instead of letting the parse fail."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text.removeprefix("json").strip()
    return text


def call_gemini(client, model_name, gen_config, prompt, max_attempts=3, backoff_seconds=2):
    """Retries on any failure. If the error message includes Google's own
    "retry in Xs" hint, waits that long instead of guessing; otherwise
    falls back to backoff_seconds * attempt.
    """
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
    """Logs one nested MLflow run per call. Swallows MLflow failures instead
    of crashing the eval loop over them -- a tracking-server hiccup
    shouldn't take down an otherwise-working generation run.

    Caller must already have pointed MLflow at a tracking server first --
    see shared/tracking.py's init_tracking(), which every notebook calls
    once near the top instead of each module here configuring its own.
    """
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
