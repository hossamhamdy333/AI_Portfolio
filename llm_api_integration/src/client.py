"""Thin wrapper around the Gemini API.

Deliberately not a class hierarchy or an abstract "LLMProvider" interface —
this project talks to exactly one provider. Adding an abstraction layer for
a single implementation is speculative generality, not good design.
"""

import logging
import os
import time

import google.generativeai as genai

logger = logging.getLogger(__name__)


def build_model(model_name: str, temperature: float, max_output_tokens: int, json_mode: bool = False):
    """Configure the SDK and return a ready-to-use GenerativeModel.

    Reads the API key from the environment rather than a config file —
    secrets and settings belong in different places so secrets never
    accidentally get committed to git.

    json_mode=True tells Gemini to return raw JSON instead of prose wrapped
    in markdown fences (```json ... ```). Without this, "respond only with
    JSON" is a suggestion the model doesn't reliably follow, and
    json.loads() breaks on the fences.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Check that .env exists and load_dotenv() "
            "ran before this call."
        )

    genai.configure(api_key=api_key)
    generation_config = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
    }
    if json_mode:
        generation_config["response_mime_type"] = "application/json"
    return genai.GenerativeModel(model_name, generation_config=generation_config)


def strip_markdown_fences(text: str) -> str:
    """Defensive fallback: strip ```json fences if the model adds them anyway.

    response_mime_type="application/json" should prevent this, but treating
    it as guaranteed rather than likely is exactly the kind of assumption
    that breaks in production. Cheap insurance.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text.removeprefix("json").strip()
    return text


def call_gemini(model, prompt: str, max_attempts: int = 3, backoff_seconds: int = 2):
    """Call Gemini with basic exponential backoff on transient failures.

    Retries exist because rate limits and timeouts are expected background
    noise for any external API call, not exceptional cases — the caller
    shouldn't have to think about this every time it makes a request.
    """
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return model.generate_content(prompt)
        except Exception as e:
            last_error = e
            logger.warning(
                f"Gemini call failed (attempt {attempt}/{max_attempts}): {e}"
            )
            if attempt < max_attempts:
                time.sleep(backoff_seconds * attempt)
    raise RuntimeError(f"Gemini call failed after {max_attempts} attempts") from last_error


def stream_gemini(model, prompt: str):
    """Yield raw response chunks as they arrive.

    Yields the chunk object itself, not chunk.text, so callers can read
    chunk.usage_metadata off the final chunk (Gemini attaches aggregate
    usage there, not on every individual chunk) for cost/token logging.
    """
    for chunk in model.generate_content(prompt, stream=True):
        yield chunk
