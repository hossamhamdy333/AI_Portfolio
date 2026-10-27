from audit import AuditLog


def test_log_writes_one_row_per_step(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl", run_id="r1")
    log.log("research", question="q")
    log.log("write", draft="d")
    rows = log.read()
    assert [r["step"] for r in rows] == ["research", "write"]
    assert rows[0]["run_id"] == "r1"
    assert rows[0]["question"] == "q"


def test_read_missing_file_returns_empty(tmp_path):
    assert AuditLog(tmp_path / "none.jsonl").read() == []


def test_log_appends_across_instances(tmp_path):
    p = tmp_path / "audit.jsonl"
    AuditLog(p, run_id="a").log("x")
    AuditLog(p, run_id="b").log("y")
    assert [r["run_id"] for r in AuditLog(p).read()] == ["a", "b"]
