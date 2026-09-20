"""
Two ways to generate a response:

  llamacpp (default) - the existing in-process llama-cpp-python setup,
                        CPU-only, loads the quantized GGUF file directly.
                        Unchanged from before this file existed.

  vllm                - an HTTP call to a running vLLM server (OpenAI-
                        compatible /v1/completions endpoint). vLLM's whole
                        reason to exist is serving many concurrent
                        requests efficiently (continuous batching,
                        PagedAttention) - the right tool specifically
                        because this is a customer-facing bot where many
                        people can be chatting at once, unlike llama.cpp
                        which serves one request at a time per process.

Both are called the same way from app.py: generate(prompt). Which one
runs is decided once, by settings.LLM_BACKEND - nothing in app.py needs
to know which backend is actually behind that call.
"""

import os

import requests

from src.config import settings

MAX_NEW_TOKENS = 100

_model = None  # only used by the llamacpp backend


def load_llamacpp_model():
    """Downloads the GGUF file once and loads it."""
    global _model
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    model_path = hf_hub_download(repo_id=settings.GGUF_REPO, filename=settings.GGUF_FILENAME)
    _model = Llama(
        model_path=model_path,
        n_ctx=1024,
        n_threads=4,        # matches the Container App's actual CPU allocation --
                             # os.cpu_count() reads the HOST's core count, not
                             # what this container is limited to, which was
                             # oversubscribing threads and slowing things down.
        n_threads_batch=4,  # threads used during prompt processing specifically
        n_batch=512,        # larger prompt-processing batch = faster prompt ingestion
        verbose=False,
    )


def _generate_llamacpp(prompt: str) -> str:
    output = _model(
        prompt,
        max_tokens=MAX_NEW_TOKENS,
        stop=["<|user|>", "<|system|>"],
        temperature=0.0,
    )
    return output["choices"][0]["text"].strip()


def _generate_llamacpp_stream(prompt: str):
    """Yields the answer piece by piece as llama.cpp produces it."""
    for chunk in _model(
        prompt,
        max_tokens=MAX_NEW_TOKENS,
        stop=["<|user|>", "<|system|>"],
        temperature=0.0,
        stream=True,
    ):
        piece = chunk["choices"][0]["text"]
        if piece:
            yield piece


def _generate_vllm(prompt: str) -> str:
    response = requests.post(
        f"{settings.VLLM_BASE_URL}/completions",
        json={
            "model": settings.VLLM_MODEL,
            "prompt": prompt,
            "max_tokens": MAX_NEW_TOKENS,
            "temperature": 0.0,
            "stop": ["<|user|>", "<|system|>"],
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["text"].strip()


def load_model():
    """Called once at app startup. The vLLM backend has nothing to load
    here - the model is already loaded in the separate vLLM server
    process, this app is just an HTTP client to it."""
    if settings.LLM_BACKEND == "llamacpp":
        load_llamacpp_model()


def is_ready() -> bool:
    if settings.LLM_BACKEND == "llamacpp":
        return _model is not None
    return True  # vllm: readiness is the remote server's problem, not this process's


def generate(prompt: str) -> str:
    if settings.LLM_BACKEND == "vllm":
        return _generate_vllm(prompt)
    return _generate_llamacpp(prompt)


def generate_stream(prompt: str):
    """Like generate(), but yields text pieces as they are produced, so the
    page can show the answer while it is still being written. The vLLM
    backend just yields its whole answer at once."""
    if settings.LLM_BACKEND == "vllm":
        yield _generate_vllm(prompt)
        return
    yield from _generate_llamacpp_stream(prompt)
