"""Tests for src/qa_generation.py -- cost math and parsing, no live API calls."""

from src.qa_generation import compute_cost_usd, strip_markdown_fences


def test_compute_cost_usd_zero_tokens():
    assert compute_cost_usd(0, 0, 0.25, 1.50) == 0.0


def test_compute_cost_usd_known_value():
    cost = compute_cost_usd(1_000_000, 1_000_000, 0.25, 1.50)
    assert round(cost, 2) == 1.75


def test_strip_markdown_fences_removes_json_fence():
    raw = """```json
{"questions": ["a", "b"]}
```"""
    cleaned = strip_markdown_fences(raw)
    assert cleaned.startswith("{")
    assert cleaned.endswith("}")


def test_strip_markdown_fences_plain_text_unchanged():
    raw = '{"questions": ["a"]}'
    assert strip_markdown_fences(raw) == raw
