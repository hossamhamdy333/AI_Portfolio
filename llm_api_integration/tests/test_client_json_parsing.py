"""Tests for strip_markdown_fences — this exists because Gemini sometimes
wraps JSON in ```json fences even in JSON mode, and json.loads/Pydantic
both choke on that if it's not stripped first.
"""

from src.client import strip_markdown_fences


def test_strips_json_fence():
    raw = '```json\n{"a": 1}\n```'
    assert strip_markdown_fences(raw) == '{"a": 1}'


def test_strips_plain_fence():
    raw = '```\n{"a": 1}\n```'
    assert strip_markdown_fences(raw) == '{"a": 1}'


def test_leaves_clean_json_untouched():
    raw = '{"a": 1}'
    assert strip_markdown_fences(raw) == '{"a": 1}'
