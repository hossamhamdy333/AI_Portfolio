"""Pydantic models that define the shape we require from LLM output.

The model produces text; text can be malformed JSON, missing fields, or the
wrong types. Validating against a schema gives a hard boundary — either the
response matches, or we get a clear error right here, not a silent bug three
functions downstream.
"""

from typing import Optional

from pydantic import BaseModel, Field


class SentimentResult(BaseModel):
    """Structured result for the /analyze endpoint."""

    sentiment: str = Field(description="One of: positive, negative, neutral")
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class ToolCallRequest(BaseModel):
    """What the model decided in response to the routing prompt.

    tool_name is optional because the routing prompt explicitly allows the
    model to answer '{"tool_name": null, "arguments": {}}' when no tool is
    needed — a non-optional str here would wrongly reject that valid case.
    arguments defaults to {} so a model response that omits the key
    entirely doesn't blow up with a missing-field error.
    """

    tool_name: Optional[str] = None
    arguments: dict = Field(default_factory=dict)
