"""
Tests for select_backend() - which functions/model name/kwargs get
picked for each backend.provider value. Pure logic, no MLflow, no live
model connections - matching this project's existing testing philosophy
of keeping logic separately testable from app.py's side effects.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from src.backend_selection import select_backend
from src.client import call_gemini, stream_gemini
from src.local_client import call_local, stream_local


BASE_CONFIG = {
    "model": {"name": "gemini-3.1-flash-lite"},
    "backend": {
        "provider": "gemini",
        "ollama": {"base_url": "http://localhost:11434/v1", "model": "llama3.1"},
        "vllm": {"base_url": "http://localhost:8001/v1", "model": "some-merged-checkpoint"},
    },
}


def _config_with_provider(provider):
    config = {**BASE_CONFIG, "backend": {**BASE_CONFIG["backend"], "provider": provider}}
    return config


def test_gemini_provider_selects_gemini_functions():
    backend = select_backend(_config_with_provider("gemini"))
    assert backend.call_model is call_gemini
    assert backend.stream_model is stream_gemini
    assert backend.model_name == "gemini-3.1-flash-lite"
    assert backend.model_kwargs == {}


def test_ollama_provider_selects_local_functions_and_its_own_config():
    backend = select_backend(_config_with_provider("ollama"))
    assert backend.call_model is call_local
    assert backend.stream_model is stream_local
    assert backend.model_name == "llama3.1"
    assert backend.model_kwargs == {"base_url": "http://localhost:11434/v1"}


def test_vllm_provider_selects_local_functions_and_its_own_config():
    backend = select_backend(_config_with_provider("vllm"))
    assert backend.call_model is call_local
    assert backend.stream_model is stream_local
    assert backend.model_name == "some-merged-checkpoint"
    assert backend.model_kwargs == {"base_url": "http://localhost:8001/v1"}


def test_ollama_and_vllm_use_the_same_functions_different_config():
    """The whole point of sharing local_client.py between them - same
    code path, only the base_url/model differ."""
    ollama_backend = select_backend(_config_with_provider("ollama"))
    vllm_backend = select_backend(_config_with_provider("vllm"))
    assert ollama_backend.call_model is vllm_backend.call_model
    assert ollama_backend.stream_model is vllm_backend.stream_model
    assert ollama_backend.model_kwargs != vllm_backend.model_kwargs


def test_unknown_provider_raises_a_clear_error():
    with pytest.raises(ValueError, match="Unknown backend.provider"):
        select_backend(_config_with_provider("some-typo"))
