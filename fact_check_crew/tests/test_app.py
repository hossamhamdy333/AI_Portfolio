import json
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

import crew as crew_module
from crew import CriticVerdict

APP = str(Path(__file__).resolve().parent.parent / "app.py")


class FakeTaskOutput:
    def __init__(self, text):
        self.output = text

    def __str__(self):
        return self.output


def setup_app(monkeypatch, tmp_path):
    fake_tools = types.ModuleType("tools")
    fake_tools.load_config = lambda path: {"revision": {"max_revisions": 1}}
    fake_tools.build_tools = lambda config: (None, None)
    monkeypatch.setitem(sys.modules, "tools", fake_tools)

    log_path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("AUDIT_LOG_PATH", str(log_path))

    def fake_initial(llm, search_tool, question):
        verdict = CriticVerdict(approved=True, feedback="fine")
        return "draft v1", verdict, FakeTaskOutput("passages"), "writer", "critic"

    monkeypatch.setattr(crew_module, "_run_initial_pass", fake_initial)
    return log_path


def get_button(at, label):
    return next(b for b in at.button if b.label == label)


def ask(at, question):
    at.text_input(key="question").set_value(question)
    get_button(at, "Run crew").click().run()


def logged_steps(log_path):
    with open(log_path, encoding="utf-8") as f:
        return [json.loads(line)["step"] for line in f if line.strip()]


def test_run_then_approve_releases_answer(monkeypatch, tmp_path):
    log_path = setup_app(monkeypatch, tmp_path)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception

    ask(at, "What is baseball?")
    assert "draft v1" in [m.value for m in at.markdown]
    assert logged_steps(log_path)[-1] == "approval_requested"

    get_button(at, "Approve and release").click().run()
    assert any("Approved" in s.value for s in at.success)
    assert logged_steps(log_path)[-2:] == ["human_decision", "final"]


def test_reject_releases_no_answer(monkeypatch, tmp_path):
    log_path = setup_app(monkeypatch, tmp_path)
    at = AppTest.from_file(APP, default_timeout=30).run()

    ask(at, "What is baseball?")
    get_button(at, "Reject").click().run()

    assert any("No answer was released" in i.value for i in at.info)
    assert logged_steps(log_path)[-1] == "final"


def test_empty_question_does_not_run_the_crew(monkeypatch, tmp_path):
    log_path = setup_app(monkeypatch, tmp_path)
    at = AppTest.from_file(APP, default_timeout=30).run()

    ask(at, "   ")

    assert any("Type a question" in w.value for w in at.warning)
    assert not log_path.exists()
