"""Tests for retry behavior in client.py.

We mock the Gemini call itself — the goal is to verify our retry loop
behaves correctly, not to test Gemini's own reliability.
"""

from unittest.mock import MagicMock

import pytest

from src.client import call_gemini


def test_call_gemini_succeeds_first_try():
    model = MagicMock()
    model.generate_content.return_value = "ok"

    result = call_gemini(model, "hello", max_attempts=3, backoff_seconds=0)

    assert result == "ok"
    assert model.generate_content.call_count == 1


def test_call_gemini_retries_then_succeeds():
    model = MagicMock()
    model.generate_content.side_effect = [Exception("timeout"), "ok"]

    result = call_gemini(model, "hello", max_attempts=3, backoff_seconds=0)

    assert result == "ok"
    assert model.generate_content.call_count == 2


def test_call_gemini_raises_after_max_attempts():
    model = MagicMock()
    model.generate_content.side_effect = Exception("always fails")

    with pytest.raises(RuntimeError):
        call_gemini(model, "hello", max_attempts=3, backoff_seconds=0)

    assert model.generate_content.call_count == 3
