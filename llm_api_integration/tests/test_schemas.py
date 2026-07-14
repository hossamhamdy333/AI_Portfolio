"""Tests confirming SentimentResult actually enforces its schema.

This is the whole point of using Pydantic here: bad shapes should fail
loudly and immediately, not pass through silently.
"""

import pytest
from pydantic import ValidationError

from src.schemas import SentimentResult, ToolCallRequest


def test_valid_sentiment_result_parses():
    raw = '{"sentiment": "positive", "confidence": 0.9, "reasoning": "upbeat tone"}'
    result = SentimentResult.model_validate_json(raw)
    assert result.sentiment == "positive"
    assert result.confidence == 0.9


def test_confidence_out_of_range_rejected():
    raw = '{"sentiment": "positive", "confidence": 1.5, "reasoning": "bad value"}'
    with pytest.raises(ValidationError):
        SentimentResult.model_validate_json(raw)


def test_missing_field_rejected():
    raw = '{"sentiment": "positive", "confidence": 0.9}'
    with pytest.raises(ValidationError):
        SentimentResult.model_validate_json(raw)


def test_valid_tool_call_parses():
    raw = '{"tool_name": "get_current_weather", "arguments": {"city": "Cairo"}}'
    result = ToolCallRequest.model_validate_json(raw)
    assert result.tool_name == "get_current_weather"
    assert result.arguments == {"city": "Cairo"}


def test_null_tool_name_is_valid():
    """The routing prompt explicitly allows this when no tool is needed —
    a non-optional schema would incorrectly reject it."""
    raw = '{"tool_name": null, "arguments": {}}'
    result = ToolCallRequest.model_validate_json(raw)
    assert result.tool_name is None


def test_missing_arguments_defaults_to_empty_dict():
    """Model sometimes omits 'arguments' entirely when no tool is needed —
    this should default, not raise a missing-field error."""
    raw = '{"tool_name": null}'
    result = ToolCallRequest.model_validate_json(raw)
    assert result.arguments == {}
