import crew as crew_module
from crew import CriticVerdict
from audit import AuditLog
from supervisor import run_supervised


class FakeTaskOutput:
    def __init__(self, text):
        self.output = text

    def __str__(self):
        return self.output


def patch_crew(monkeypatch, first_approved, second_approved=True):
    def fake_initial(llm, search_tool, question):
        verdict = CriticVerdict(approved=first_approved, feedback="first feedback")
        return "draft v1", verdict, FakeTaskOutput("passages"), "writer", "critic"

    def fake_revision(llm, question, feedback, research_task, writer, critic):
        return "draft v2", CriticVerdict(approved=second_approved, feedback="second feedback")

    monkeypatch.setattr(crew_module, "_run_initial_pass", fake_initial)
    monkeypatch.setattr(crew_module, "_run_revision_pass", fake_revision)


def steps(audit):
    return [r["step"] for r in audit.read()]


def test_approved_first_pass_goes_to_human(monkeypatch, tmp_path):
    patch_crew(monkeypatch, first_approved=True)
    audit = AuditLog(tmp_path / "a.jsonl")
    result = run_supervised(None, None, "q", audit, lambda d, v: (True, "ok"))

    assert result["status"] == "approved"
    assert result["answer"] == "draft v1"
    assert result["n_revisions"] == 0
    assert steps(audit) == [
        "start", "research", "write", "critique", "decision",
        "approval_requested", "human_decision", "final",
    ]


def test_rejected_draft_is_revised_once(monkeypatch, tmp_path):
    patch_crew(monkeypatch, first_approved=False, second_approved=True)
    audit = AuditLog(tmp_path / "a.jsonl")
    result = run_supervised(None, None, "q", audit, lambda d, v: (True, ""))

    assert result["n_revisions"] == 1
    assert result["answer"] == "draft v2"
    assert steps(audit).count("write") == 2
    assert steps(audit).count("critique") == 2
    actions = [r["action"] for r in audit.read() if r["step"] == "decision"]
    assert actions == ["revise", "send_to_human"]


def test_human_rejection_returns_no_answer(monkeypatch, tmp_path):
    patch_crew(monkeypatch, first_approved=True)
    audit = AuditLog(tmp_path / "a.jsonl")
    result = run_supervised(None, None, "q", audit, lambda d, v: (False, "not good"))

    assert result["status"] == "rejected"
    assert result["answer"] is None
    rows = audit.read()
    assert rows[-2]["step"] == "human_decision"
    assert rows[-2]["note"] == "not good"
    assert rows[-1]["status"] == "rejected"


def test_human_sees_critic_verdict_when_still_rejected(monkeypatch, tmp_path):
    patch_crew(monkeypatch, first_approved=False)
    seen = {}

    def approve(draft, verdict):
        seen["draft"] = draft
        seen["approved"] = verdict.approved
        return False, ""

    audit = AuditLog(tmp_path / "a.jsonl")
    result = run_supervised(None, None, "q", audit, approve, max_revisions=0)

    assert seen == {"draft": "draft v1", "approved": False}
    assert result["n_revisions"] == 0
    assert result["critic_approved"] is False
