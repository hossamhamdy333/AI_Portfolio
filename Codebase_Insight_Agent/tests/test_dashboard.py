import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tracing


def span(name, start, duration, error=False, **attrs):
    return {"name": name, "start": start, "duration_s": duration, "error": error, "attributes": attrs}


def test_summary_counts_times_errors_and_retries():
    spans = [
        span("ask", 1, 2.0, retries=0, projects="a"),
        span("ask", 2, 4.0, retries=2, projects="b"),
        span("ask", 3, 1.0, error=True),
        span("ask", 4, 3.0, retries=0, projects="a,b"),
    ]
    out = tracing.summarize(spans)
    assert out["questions"] == 4
    assert out["failed"] == 1
    assert out["error_rate"] == 0.25
    assert out["retried"] == 1
    assert out["avg_s"] == 2.5
    assert out["max_s"] == 4.0
    assert out["p95_s"] == 4.0
    assert out["slowest"][0]["projects"] == "b"


def test_summary_cost_comes_from_tokens_and_prices():
    spans = [
        span("ask", 1, 1.0),
        span("draft", 1, 0.5, input_tokens=2_000_000, output_tokens=1_000_000),
        span("critique", 1, 0.5, input_tokens=1_000_000, output_tokens=0),
    ]
    out = tracing.summarize(spans, input_per_m=0.30, output_per_m=2.50)
    assert out["input_tokens"] == 3_000_000
    assert out["output_tokens"] == 1_000_000
    assert out["est_cost_usd"] == 3.4
    assert out["cost_per_question_usd"] == 3.4


def test_summary_per_step_table_is_in_pipeline_order():
    spans = [span("critique", 1, 1.0), span("zzz", 1, 1.0), span("ask", 1, 1.0), span("draft", 1, 1.0)]
    names = [row["name"] for row in tracing.summarize(spans)["spans"]]
    assert names == ["ask", "draft", "critique", "zzz"]


def test_summary_of_nothing_does_not_divide_by_zero():
    out = tracing.summarize([])
    assert out["questions"] == 0
    assert out["error_rate"] == 0.0
    assert out["avg_s"] == 0.0
    assert out["est_cost_usd"] == 0.0
    assert out["recent"] == []


def test_recent_keeps_only_the_latest_questions_in_time_order():
    spans = [span("ask", i, float(i)) for i in range(10, 0, -1)]
    recent = tracing.summarize(spans, recent=3)["recent"]
    assert [r["start"] for r in recent] == [8, 9, 10]


def test_read_spans_skips_bad_lines_and_missing_files(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps(span("ask", 1, 1.0)) + "\nnot json\n\n" + json.dumps(span("ask", 2, 1.0)) + "\n")
    assert len(tracing.read_spans(str(path))) == 2
    assert tracing.read_spans(str(tmp_path / "missing.jsonl")) == []


def test_read_spans_only_reads_the_tail(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text("".join(json.dumps(span("ask", i, 1.0)) + "\n" for i in range(50)))
    spans = tracing.read_spans(str(path), max_lines=10)
    assert [s["start"] for s in spans] == list(range(40, 50))
