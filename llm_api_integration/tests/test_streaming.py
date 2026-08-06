"""Tests for stream_gemini yielding raw chunks (needed so app.py can read
chunk.usage_metadata off the final chunk for cost logging).
"""

from unittest.mock import MagicMock

from src.client import stream_gemini


def test_stream_gemini_yields_raw_chunks_not_text():
    fake_chunk = MagicMock()
    fake_chunk.text = "hello"
    model = MagicMock()
    model.generate_content.return_value = [fake_chunk]

    chunks = list(stream_gemini(model, "prompt"))

    assert chunks == [fake_chunk]
    assert chunks[0].text == "hello"
    model.generate_content.assert_called_once_with("prompt", stream=True)
