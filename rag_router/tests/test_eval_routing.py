"""Tests for eval_routing.py -- no live API calls, no live Qdrant.

These are the same three behaviors manually verified while building this
module (deterministic-error short-circuit, rate-limit retry still works,
per-selector checkpoint isolation), formalized as regression tests.
"""

import json
import os

import pandas as pd
import pytest

from eval_routing import _is_selector_parsing_bug, _route_with_retry, run_eval


def test_is_selector_parsing_bug_matches_known_signature():
    assert _is_selector_parsing_bug(TypeError("unsupported operand type(s) for -: 'NoneType' and 'int'"))


def test_is_selector_parsing_bug_rejects_unrelated_type_error():
    assert not _is_selector_parsing_bug(TypeError("some unrelated type error"))


def test_is_selector_parsing_bug_rejects_non_type_error():
    assert not _is_selector_parsing_bug(ValueError("NoneType and int"))


def test_route_with_retry_short_circuits_on_deterministic_bug(monkeypatch):
    call_count = {"n": 0}

    def fake_route_and_query(router, domains, question):
        call_count["n"] += 1
        raise TypeError("unsupported operand type(s) for -: 'NoneType' and 'int'")

    import router as router_module
    monkeypatch.setattr(router_module, "route_and_query", fake_route_and_query)

    result = _route_with_retry(None, ["sports"], "q", max_attempts=5)
    assert call_count["n"] == 1  # not retried 5 times for a deterministic failure
    assert result[1] is None  # routed_domain is None on failure


def test_route_with_retry_retries_transient_errors(monkeypatch):
    call_count = {"n": 0}

    def fake_route_and_query(router, domains, question):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise Exception("429 RESOURCE_EXHAUSTED retry in 0.01s")
        return ("answer", "sports", [])

    import router as router_module
    monkeypatch.setattr(router_module, "route_and_query", fake_route_and_query)

    result = _route_with_retry(None, ["sports"], "q", max_attempts=5)
    assert call_count["n"] == 3
    assert result[1] == "sports"


def test_run_eval_uses_separate_checkpoints_per_selector(tmp_path, monkeypatch):
    import mlflow
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/test_mlflow.db")
    mlflow.set_experiment("test")

    def fake_route_and_query(router, domains, question):
        return (f"answer for {question}", "sports", [])

    import router as router_module
    monkeypatch.setattr(router_module, "route_and_query", fake_route_and_query)

    qa_df = pd.DataFrame([
        {"question": f"q{i}", "answer": "a", "article_id": f"id{i}", "article_title": "sports_0", "domain": "sports"}
        for i in range(3)
    ])

    cp_embedding = str(tmp_path / "embedding.json")
    cp_llm = str(tmp_path / "llm.json")

    run_eval(None, ["sports"], qa_df, selector_name="embedding", checkpoint_path=cp_embedding, sleep_seconds=0)
    run_eval(None, ["sports"], qa_df, selector_name="llm", checkpoint_path=cp_llm, sleep_seconds=0)

    assert os.path.exists(cp_embedding)
    assert os.path.exists(cp_llm)
    with open(cp_embedding) as f:
        assert len(json.load(f)) == 3
    with open(cp_llm) as f:
        assert len(json.load(f)) == 3
