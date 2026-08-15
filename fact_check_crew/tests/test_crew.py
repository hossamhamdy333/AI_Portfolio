"""Tests for crew.py's retry-loop control flow -- mocks the actual
CrewAI-calling functions (_run_initial_pass, _run_revision_pass) so these
test only the loop logic itself: does it skip revision when approved
immediately, does it pass feedback through correctly, does it stop at
max_revisions instead of looping forever. No live LLM calls.
"""

import crew as crew_module
from crew import run_crew_with_revision, CriticVerdict


class FakeTaskOutput:
    """Stands in for a CrewAI Task's .output after kickoff()."""
    def __init__(self, text):
        self.output = text

    def __str__(self):
        return self.output


def test_approved_immediately_skips_revision(monkeypatch):
    call_log = []

    def fake_initial(llm, search_tool, question):
        call_log.append("initial")
        return "draft v1", CriticVerdict(approved=True, feedback="looks good"), FakeTaskOutput("passages"), "writer", "critic"

    def fake_revision(*a, **kw):
        call_log.append("revision")
        raise AssertionError("revision should not run when approved on the first pass")

    monkeypatch.setattr(crew_module, "_run_initial_pass", fake_initial)
    monkeypatch.setattr(crew_module, "_run_revision_pass", fake_revision)

    result = run_crew_with_revision(None, None, "question", max_revisions=1)

    assert result["approved"] is True
    assert result["n_revisions"] == 0
    assert result["answer"] == "draft v1"
    assert call_log == ["initial"]


def test_rejected_then_approved_runs_exactly_one_revision(monkeypatch):
    call_log = []

    def fake_initial(llm, search_tool, question):
        call_log.append("initial")
        return "draft v1", CriticVerdict(approved=False, feedback="unsupported claim X"), FakeTaskOutput("real passages"), "writer", "critic"

    def fake_revision(llm, question, feedback, research_task, writer, critic):
        call_log.append("revision")
        assert feedback == "unsupported claim X"
        assert str(research_task.output) == "real passages"  # revision reuses the ORIGINAL research, doesn't re-search
        return "draft v2 (revised)", CriticVerdict(approved=True, feedback="now supported")

    monkeypatch.setattr(crew_module, "_run_initial_pass", fake_initial)
    monkeypatch.setattr(crew_module, "_run_revision_pass", fake_revision)

    result = run_crew_with_revision(None, None, "question", max_revisions=1)

    assert result["approved"] is True
    assert result["n_revisions"] == 1
    assert result["answer"] == "draft v2 (revised)"
    assert result["research_passages"] == "real passages"
    assert call_log == ["initial", "revision"]


def test_still_rejected_stops_at_max_revisions(monkeypatch):
    call_log = []

    def fake_initial(llm, search_tool, question):
        call_log.append("initial")
        return "draft v1", CriticVerdict(approved=False, feedback="bad claim"), FakeTaskOutput("passages"), "writer", "critic"

    def fake_revision(llm, question, feedback, research_task, writer, critic):
        call_log.append("revision")
        return "draft v2 still bad", CriticVerdict(approved=False, feedback="still bad")

    monkeypatch.setattr(crew_module, "_run_initial_pass", fake_initial)
    monkeypatch.setattr(crew_module, "_run_revision_pass", fake_revision)

    result = run_crew_with_revision(None, None, "question", max_revisions=1)

    # Not approved, but critically: only ONE revision attempt, not an
    # open-ended retry loop.
    assert result["approved"] is False
    assert result["n_revisions"] == 1
    assert call_log == ["initial", "revision"]


def test_max_revisions_zero_never_revises(monkeypatch):
    call_log = []

    def fake_initial(llm, search_tool, question):
        call_log.append("initial")
        return "draft", CriticVerdict(approved=False, feedback="bad"), FakeTaskOutput("passages"), "writer", "critic"

    def fake_revision(*a, **kw):
        call_log.append("revision")
        raise AssertionError("should never be called when max_revisions=0")

    monkeypatch.setattr(crew_module, "_run_initial_pass", fake_initial)
    monkeypatch.setattr(crew_module, "_run_revision_pass", fake_revision)

    result = run_crew_with_revision(None, None, "question", max_revisions=0)

    assert result["n_revisions"] == 0
    assert call_log == ["initial"]


def test_build_agents_returns_three_distinct_roles():
    from crew import build_agents
    from crewai import LLM
    from crewai.tools import tool

    @tool("fake_tool")
    def fake_tool(query: str) -> str:
        """A fake tool for testing."""
        return "result"

    llm = LLM(model="gemini/gemini-3.1-flash-lite", api_key="fake-key")
    researcher, writer, critic = build_agents(llm, fake_tool)

    assert {researcher.role, writer.role, critic.role} == {"Researcher", "Writer", "Critic"}
    assert fake_tool in researcher.tools
    assert writer.tools == [] or writer.tools is None
