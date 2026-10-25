import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tracing


def read_spans(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_spans_are_written_with_parent_and_tokens(tmp_path, monkeypatch):
    path = tmp_path / "traces.jsonl"
    monkeypatch.setattr(tracing._provider._active_span_processor._span_processors[0].span_exporter, "path", str(path))

    class Reply:
        usage_metadata = {"input_tokens": 12, "output_tokens": 5}

    with tracing.span("ask"):
        with tracing.span("draft") as s:
            s.set_attributes(tracing.token_counts(Reply()))

    spans = {s["name"]: s for s in read_spans(path)}
    assert spans["draft"]["parent_id"] == spans["ask"]["span_id"]
    assert spans["draft"]["trace_id"] == spans["ask"]["trace_id"]
    assert spans["draft"]["attributes"]["input_tokens"] == 12
    assert spans["draft"]["attributes"]["output_tokens"] == 5
    assert spans["ask"]["error"] is False


def test_an_exception_marks_the_span_as_error(tmp_path, monkeypatch):
    path = tmp_path / "traces.jsonl"
    monkeypatch.setattr(tracing._provider._active_span_processor._span_processors[0].span_exporter, "path", str(path))

    try:
        with tracing.span("ask"):
            raise ValueError("boom")
    except ValueError:
        pass

    assert read_spans(path)[0]["error"] is True


def test_token_counts_default_to_zero_when_missing():
    assert tracing.token_counts(object()) == {"input_tokens": 0, "output_tokens": 0}


def test_unwritable_path_does_not_break_the_app(monkeypatch):
    monkeypatch.setattr(tracing._provider._active_span_processor._span_processors[0].span_exporter, "path", "/no/such/folder/t.jsonl")
    with tracing.span("ask"):
        pass
