import uuid

import pytest
from langgraph.checkpoint.memory import MemorySaver

from agents.graph import build_graph


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Returns canned replies keyed by a substring of the prompt, in order,
    so each test controls exactly what each node 'thinks'."""

    def __init__(self, replies_by_keyword):
        self.replies_by_keyword = replies_by_keyword
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        for keyword, reply in self.replies_by_keyword.items():
            if keyword in prompt:
                return FakeMessage(reply)
        raise AssertionError(f"No fake reply configured for prompt: {prompt[:80]!r}")


def _init_state(task="test task"):
    return {
        "task": task, "plan": "", "research": "", "analysis": "",
        "report": "", "needs_human_review": False, "review_note": None,
    }


def test_graph_runs_end_to_end_without_conflict(monkeypatch):
    monkeypatch.setattr("agents.graph.web_search", lambda q: "web info")
    monkeypatch.setattr("agents.graph.retrieve", lambda client, q: "doc info")

    llm = FakeLLM({
        "Break this task": "1. Research X\n2. Summarize",
        "conflict": "NO",
        "decide if a short Python": "NO_CODE_NEEDED",
        "Write a concise Markdown": "# Report\nFinal answer.",
    })
    graph = build_graph(llm, qdrant_client=object(), checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    result = graph.invoke(_init_state(), config)
    assert result["needs_human_review"] is False

    # The graph unconditionally pauses before "analyst" (interrupt_before=["analyst"]);
    # it's the caller's job (the API) to auto-resume when no conflict was flagged.
    result = graph.invoke(None, config)
    assert result["report"] == "# Report\nFinal answer."
    assert result["analysis"] == "No code analysis needed."


def test_graph_pauses_for_human_review_on_conflict(monkeypatch):
    monkeypatch.setattr("agents.graph.web_search", lambda q: "web info: X = 10")
    monkeypatch.setattr("agents.graph.retrieve", lambda client, q: "doc info: X = 20")

    llm = FakeLLM({
        "Break this task": "1. Research X",
        "conflict": "YES",
    })
    graph = build_graph(llm, qdrant_client=object(), checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    result = graph.invoke(_init_state(), config)

    # Graph should stop before "analyst" and flag the conflict, not silently
    # continue to write a report from contradictory sources.
    assert result["needs_human_review"] is True
    assert result["review_note"] == "Conflicting information detected between sources."
    assert "report" not in result or not result.get("report")


def test_graph_resumes_after_human_approval(monkeypatch):
    monkeypatch.setattr("agents.graph.web_search", lambda q: "web info")
    monkeypatch.setattr("agents.graph.retrieve", lambda client, q: "doc info")

    llm = FakeLLM({
        "Break this task": "1. Research X",
        "conflict": "YES",
        "decide if a short Python": "NO_CODE_NEEDED",
        "Write a concise Markdown": "# Resumed report",
    })
    graph = build_graph(llm, qdrant_client=object(), checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    paused = graph.invoke(_init_state(), config)
    assert paused["needs_human_review"] is True

    # Resuming (like /resume/{thread_id} does in the API) should push the run
    # through analyst -> writer to completion.
    resumed = graph.invoke(None, config)
    assert resumed["report"] == "# Resumed report"


def test_graph_runs_generated_code_when_analyst_requests_it(monkeypatch):
    monkeypatch.setattr("agents.graph.web_search", lambda q: "web info")
    monkeypatch.setattr("agents.graph.retrieve", lambda client, q: "doc info")

    llm = FakeLLM({
        "Break this task": "1. Compute something",
        "conflict": "NO",
        "decide if a short Python": "print(2 + 2)",
        "Write a concise Markdown": "# Report with analysis",
    })
    graph = build_graph(llm, qdrant_client=object(), checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    graph.invoke(_init_state(), config)  # pauses before "analyst"
    result = graph.invoke(None, config)  # resume, like the API does when no conflict

    assert "4" in result["analysis"]
    assert result["report"] == "# Report with analysis"
