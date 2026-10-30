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


def logged_steps(log_path):
    with open(log_path, encoding="utf-8") as f:
        return [json.loads(line)["step"] for line in f if line.strip()]


def test_run_then_approve_releases_answer(monkeypatch, tmp_path):
    log_path = setup_app(monkeypatch, tmp_path)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception

    get_button(at, "Run crew").click().run()
    assert "draft v1" in [m.value for m in at.markdown]
    assert logged_steps(log_path)[-1] == "approval_requested"

    get_button(at, "Approve").click().run()
    assert any("approved" in s.value for s in at.subheader)
    assert logged_steps(log_path)[-2:] == ["human_decision", "final"]


def test_reject_releases_no_answer(monkeypatch, tmp_path):
    log_path = setup_app(monkeypatch, tmp_path)
    at = AppTest.from_file(APP, default_timeout=30).run()

    get_button(at, "Run crew").click().run()
    get_button(at, "Reject").click().run()

    assert any("rejected" in s.value for s in at.subheader)
    assert "No answer released." in [m.value for m in at.markdown]
    assert logged_steps(log_path)[-1] == "final"
