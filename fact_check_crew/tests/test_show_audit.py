from audit import AuditLog
from show_audit import load_runs, format_run


def test_load_runs_groups_rows_by_run_id(tmp_path):
    p = tmp_path / "a.jsonl"
    AuditLog(p, run_id="a").log("start", question="q1")
    AuditLog(p, run_id="b").log("start", question="q2")
    AuditLog(p, run_id="a").log("final", status="approved")

    runs = load_runs(p)
    assert list(runs) == ["a", "b"]
    assert [r["step"] for r in runs["a"]] == ["start", "final"]


def test_load_runs_missing_file_is_empty(tmp_path):
    assert load_runs(tmp_path / "none.jsonl") == {}


def test_format_run_shows_step_and_fields_and_cuts_long_values(tmp_path):
    p = tmp_path / "a.jsonl"
    log = AuditLog(p, run_id="a")
    log.log("write", attempt=0, draft="x" * 500)

    out = format_run(load_runs(p)["a"], width=20)
    assert "write" in out
    assert "attempt=0" in out
    assert "draft=" + "x" * 20 in out
    assert "x" * 21 not in out
