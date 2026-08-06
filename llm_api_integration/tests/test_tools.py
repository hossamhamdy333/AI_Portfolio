"""Tests confirming the tool dispatcher calls the right function
and rejects unknown tool names instead of failing silently.
"""

import pytest

from src.tools import run_tool


def test_run_tool_weather():
    result = run_tool("get_current_weather", {"city": "Cairo"})
    assert result["city"] == "Cairo"
    assert "temp_c" in result


def test_run_tool_search_finds_match():
    result = run_tool("search_documents", {"query": "structured"})
    assert result[0]["id"] == 3


def test_run_tool_unknown_name_raises():
    with pytest.raises(ValueError):
        run_tool("not_a_real_tool", {})
