"""Client for a self-hosted, OpenAI-compatible backend - Ollama or vLLM.

One module for both, not two, because they're the same wire protocol:
Ollama and vLLM both expose an OpenAI-compatible `/v1/chat/completions`
endpoint. The only difference between them from this code's point of
view is which base_url and model name config.yaml points at - the HTTP
shape is identical either way.

Mirrors client.py's function names and shapes on purpose (build_model /
call_local / stream_local, matching build_model / call_gemini /
stream_gemini) so app.py can pick a backend with one `if`, not a rewrite.
The response objects returned here duck-type Gemini's shape too
(`.text`, `.usage_metadata.prompt_token_count`,
`.usage_metadata.candidates_token_count`) so tracking.py's
extract_token_counts() and every route in app.py that reads
`response.text` work completely unchanged, regardless of which backend
actually served the request.
"""

import json
import logging
import time
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)


@dataclass
class LocalModelConfig:
    """What build_model() returns here, instead of a live SDK object like
    Gemini's GenerativeModel - there's nothing to hold open, an
    OpenAI-compatible endpoint is just plain HTTP calls, so this is
    really just the settings each call needs."""
    base_url: str
    model_name: str
    temperature: float
    max_output_tokens: int
    json_mode: bool = False


class _UsageMetadata:
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_token_count = prompt_tokens
        self.candidates_token_count = completion_tokens


class _LocalResponse:
    """Duck-types a Gemini response closely enough that existing code
    (extract_token_counts, every `response.text` read in app.py) doesn't
    need to know or care which backend actually produced it."""
    def __init__(self, text: str, prompt_tokens: int = 0, completion_tokens: int = 0):
        self.text = text
        self.usage_metadata = _UsageMetadata(prompt_tokens, completion_tokens)


def build_model(model_name: str, temperature: float, max_output_tokens: int,
                 json_mode: bool = False, base_url: str = "http://localhost:11434/v1") -> LocalModelConfig:
    return LocalModelConfig(
        base_url=base_url,
        model_name=model_name,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        json_mode=json_mode,
    )


def call_local(model: LocalModelConfig, prompt: str, max_attempts: int = 3, backoff_seconds: int = 2) -> _LocalResponse:
    """Same retry/backoff shape as client.py's call_gemini - rate limits
    and timeouts are expected background noise for any HTTP call to a
    model server, self-hosted or not."""
    payload = {
        "model": model.model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": model.temperature,
        "max_tokens": model.max_output_tokens,
    }
    if model.json_mode:
        payload["response_format"] = {"type": "json_object"}

    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(f"{model.base_url}/chat/completions", json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return _LocalResponse(
                text=text,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            )
        except Exception as e:
            last_error = e
            logger.warning(f"Local model call failed (attempt {attempt}/{max_attempts}): {e}")
            if attempt < max_attempts:
                time.sleep(backoff_seconds * attempt)
    raise RuntimeError(f"Local model call failed after {max_attempts} attempts") from last_error


def stream_local(model: LocalModelConfig, prompt: str):
    """Yields the same kind of chunk object stream_gemini yields - each
    with a `.text` for the token that just arrived, and the final chunk
    additionally carrying `.usage_metadata` (stream_options.include_usage
    is what makes an OpenAI-compatible server attach usage to the last
    chunk instead of not sending it at all in streaming mode)."""
    payload = {
        "model": model.model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": model.temperature,
        "max_tokens": model.max_output_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    with requests.post(f"{model.base_url}/chat/completions", json=payload, stream=True, timeout=60) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[len("data: "):]
            if data_str.strip() == "[DONE]":
                return

            chunk = json.loads(data_str)
            usage = chunk.get("usage")
            if usage:
                yield _LocalResponse(text="", prompt_tokens=usage.get("prompt_tokens", 0), completion_tokens=usage.get("completion_tokens", 0))
                continue

            delta = chunk["choices"][0]["delta"]
            token_text = delta.get("content", "")
            if token_text:
                yield _LocalResponse(text=token_text)
