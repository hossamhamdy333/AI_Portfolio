"""Picks which backend serves requests, based on config.yaml's backend.provider.

Pulled out of app.py on purpose, matching this project's existing
convention (see app.py's own docstring: routes only, logic lives in
plain, separately-testable modules) - app.py already has import-time
side effects (init_tracking() hits MLflow, build_model() can make a live
connection), so testing this selection logic would otherwise mean
triggering all of that just to check an if-statement.
"""

from dataclasses import dataclass
from typing import Callable

from src.client import build_model as build_gemini_model, call_gemini, stream_gemini
from src.local_client import build_model as build_local_model, call_local, stream_local


@dataclass
class Backend:
    build_model_fn: Callable
    call_model: Callable
    stream_model: Callable
    model_name: str
    model_kwargs: dict


def select_backend(config: dict) -> Backend:
    provider = config["backend"]["provider"]

    if provider == "gemini":
        return Backend(
            build_model_fn=build_gemini_model,
            call_model=call_gemini,
            stream_model=stream_gemini,
            model_name=config["model"]["name"],
            model_kwargs={},
        )

    if provider in ("ollama", "vllm"):
        backend_config = config["backend"][provider]
        return Backend(
            build_model_fn=build_local_model,
            call_model=call_local,
            stream_model=stream_local,
            model_name=backend_config["model"],
            model_kwargs={"base_url": backend_config["base_url"]},
        )

    raise ValueError(f"Unknown backend.provider in config.yaml: {provider!r} (expected gemini, ollama, or vllm)")
