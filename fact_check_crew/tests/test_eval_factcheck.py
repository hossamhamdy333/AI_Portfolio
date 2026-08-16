"""Tests for eval_factcheck.py -- no live LLM calls, everything mocked."""

from eval_factcheck import judge_faithfulness, summarize, run_comparison


class FakeLLM:
    def __init__(self, response_text):
        self.response_text = response_text

    def call(self, prompt):
        return self.response_text


def test_judge_faithfulness_parses_valid_json():
    llm = FakeLLM('{"unsupported_claims": ["claim A"]}')
    assert judge_faithfulness(llm, "answer", "passages") == ["claim A"]


def test_judge_faithfulness_empty_list_means_fully_supported():
    llm = FakeLLM('{"unsupported_claims": []}')
    assert judge_faithfulness(llm, "answer", "passages") == []


def test_judge_faithfulness_strips_markdown_fences():
    llm = FakeLLM('```json\n{"unsupported_claims": ["x"]}\n```')
    assert judge_faithfulness(llm, "answer", "passages") == ["x"]


def test_judge_faithfulness_malformed_json_returns_none_not_empty():
    # Critical distinction: a judge that failed to respond usefully must
    # not be silently counted as "found nothing wrong" -- that would
    # quietly inflate whichever condition happens to trigger more
    # parsing failures, making it look more faithful than it is.
    llm = FakeLLM("not valid json")
    assert judge_faithfulness(llm, "answer", "passages") is None


def test_summarize_excludes_judge_failures_from_rate():
    results = [
        {"baseline_unsupported_claims": ["x"], "crew_unsupported_claims": [], "crew_n_revisions": 1, "crew_approved": True},
        {"baseline_unsupported_claims": [], "crew_unsupported_claims": [], "crew_n_revisions": 0, "crew_approved": True},
        {"baseline_unsupported_claims": None, "crew_unsupported_claims": ["y"], "crew_n_revisions": 1, "crew_approved": False},
    ]
    summary = summarize(results)
    assert summary["baseline_hallucination_rate"] == 0.5  # 1 of 2 judged (the None is excluded)
    assert summary["baseline_judge_failures"] == 1
    assert summary["crew_hallucination_rate"] == 1 / 3
    assert summary["crew_judge_failures"] == 0
    assert summary["crew_avg_revisions"] == 2 / 3
    assert summary["crew_approval_rate"] == 2 / 3


def test_summarize_empty_results():
    summary = summarize([])
    assert summary["n_questions"] == 0
    assert summary["n_failed"] == 0
    assert summary["baseline_hallucination_rate"] is None
    assert summary["crew_avg_revisions"] is None


def test_summarize_excludes_failed_questions_from_all_rates():
    results = [
        {"baseline_unsupported_claims": [], "crew_unsupported_claims": [], "crew_n_revisions": 1, "crew_approved": True, "failed": False},
        {"baseline_unsupported_claims": None, "crew_unsupported_claims": None, "crew_n_revisions": 0, "crew_approved": False, "failed": True},
    ]
    summary = summarize(results)
    assert summary["n_questions"] == 2
    assert summary["n_failed"] == 1
    # The failed row must not count as "0 revisions, rejected" -- it never
    # really ran the crew, so it's excluded entirely, not counted against it.
    assert summary["crew_avg_revisions"] == 1.0
    assert summary["crew_approval_rate"] == 1.0


def test_run_comparison_checkpoints_and_resumes(tmp_path):
    import pandas as pd

    class FakeSearchTool:
        def run(self, query):
            return "passages"

    class FakeLLM:
        def call(self, prompt):
            return '{"unsupported_claims": []}' if "List every claim" in prompt else "answer"

    def fake_crew(llm, search_tool, question, max_revisions):
        return {"answer": "crew answer", "approved": True, "n_revisions": 0, "research_passages": "passages"}

    qa_df = pd.DataFrame([{"question": f"q{i}"} for i in range(3)])
    checkpoint_path = str(tmp_path / "checkpoint.json")

    results = run_comparison(FakeLLM(), FakeSearchTool(), fake_crew, qa_df,
                              checkpoint_path=checkpoint_path, checkpoint_every=1, sleep_seconds=0)
    assert len(results) == 3

    # Rerunning against the same checkpoint should resume (no new work),
    # not redo everything from scratch.
    results_again = run_comparison(FakeLLM(), FakeSearchTool(), fake_crew, qa_df,
                                    checkpoint_path=checkpoint_path, checkpoint_every=1, sleep_seconds=0)
    assert len(results_again) == 3


def test_run_comparison_logs_failure_instead_of_crashing():
    import pandas as pd

    class FakeSearchTool:
        def run(self, query):
            return "passages"

    class FakeLLM:
        def call(self, prompt):
            return '{"unsupported_claims": []}' if "List every claim" in prompt else "answer"

    def fake_crew_always_fails(llm, search_tool, question, max_revisions):
        raise RuntimeError("simulated failure")

    qa_df = pd.DataFrame([{"question": "q1"}, {"question": "q2"}])
    results = run_comparison(FakeLLM(), FakeSearchTool(), fake_crew_always_fails, qa_df, sleep_seconds=0)

    # Both questions logged as failed, not a crashed/incomplete run.
    assert len(results) == 2
    assert all(r["failed"] for r in results)


def test_run_comparison_judges_crew_against_its_own_passages():
    import pandas as pd

    class FakeSearchTool:
        def run(self, query):
            return "baseline passages for: " + query

    def fake_run_crew(llm, search_tool, question, max_revisions):
        return {
            "answer": "crew answer",
            "approved": True,
            "n_revisions": 0,
            "final_feedback": "ok",
            "research_passages": "crew's OWN passages, different from baseline's",
        }

    seen_passages = []

    class RecordingLLM:
        def call(self, prompt):
            seen_passages.append(prompt)
            if "unsupported_claims" in prompt.lower() or "List every claim" in prompt:
                return '{"unsupported_claims": []}'
            return "some answer"

    qa_df = pd.DataFrame([{"question": "what is football?"}])
    results = run_comparison(RecordingLLM(), FakeSearchTool(), fake_run_crew, qa_df, max_revisions=1, sleep_seconds=0)

    assert len(results) == 1
    # Confirms the crew's answer got judged against ITS OWN research_passages,
    # not the baseline's separately-fetched ones -- the actual bug caught
    # and fixed while designing this module.
    judge_prompts_for_crew = [p for p in seen_passages if "crew's OWN passages" in p]
    assert len(judge_prompts_for_crew) >= 1
