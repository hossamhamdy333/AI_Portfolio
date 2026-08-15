"""Comparison eval: does the research -> write -> critique loop catch
unsupported claims a single-pass answer wouldn't?

Both conditions are scored by the same judge function
(judge_faithfulness), and each is judged against the passages it actually
used, not a shared/assumed passage set -- see crew.py's
run_crew_with_revision docstring for why that distinction matters here.
"""

import json
import os
import time


def run_baseline(llm, search_tool, question):
    """Single pass: search once, answer directly from what's found -- no
    writer/critic separation, no revision. This is the comparison point
    the crew needs to actually beat, not a strawman.
    """
    passages = search_tool.run(query=question)
    prompt = f"""Answer the question using only the passages below. If the passages don't contain the answer, say so plainly instead of guessing.

Passages:
{passages}

Question: {question}

Answer:"""
    response = llm.call(prompt)
    return str(response), passages


FAITHFULNESS_PROMPT = """You are checking whether an answer's claims are actually supported by the given source passages.

Passages:
{passages}

Answer to check:
{answer}

List every claim in the answer that is NOT supported by the passages above. If every claim is supported, return an empty list.

Respond ONLY with valid JSON, no other text: {{"unsupported_claims": ["claim 1", "claim 2"]}}"""


def judge_faithfulness(llm, answer, passages):
    """One direct LLM call, used identically for the baseline and the
    crew's final answer -- same judge, same prompt shape, so neither
    condition gets an easier or harder check than the other.

    Returns None (not an empty list) if the judge's response wasn't
    parseable JSON -- callers must not treat a failed judgment as "zero
    unsupported claims found", since those mean very different things for
    a hallucination-rate metric.
    """
    prompt = FAITHFULNESS_PROMPT.format(passages=passages, answer=answer)
    response = llm.call(prompt)
    text = str(response).strip()
    if text.startswith("```"):
        text = text.split("```")[1].removeprefix("json").strip()
    try:
        parsed = json.loads(text)
        return parsed.get("unsupported_claims", [])
    except json.JSONDecodeError:
        return None


def run_comparison(llm, search_tool, run_crew_fn, qa_df, max_revisions=1, sleep_seconds=8,
                    checkpoint_path=None, checkpoint_every=5):
    """qa_df needs a 'question' column (rows from rag_router's synthetic
    eval set -- see notebooks/02_evaluation.ipynb). run_crew_fn: injected
    so this can be tested against a fake instead of a live crew, same
    pattern as crew.py's own retry-loop split.

    checkpoint_path: if given, progress is saved as JSON every
    checkpoint_every questions, and resumed from there on rerun. Without
    this, a single failed question (or hitting Gemini's free-tier daily
    quota partway through -- 100 questions here means ~600-800 calls,
    comfortably over the ~500/day free-tier ceiling this project family
    has already hit twice) loses every question scored so far. A failed
    question is logged and counted (via *_failed fields, excluded from
    summarize()'s hallucination rate the same way a None judge result
    is), not allowed to crash the run.
    """
    results = []
    already_done_questions = set()

    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            results = json.load(f)
        already_done_questions = {r["question"] for r in results}
        print(f"Resuming: {len(results)}/{len(qa_df)} questions already done")

    for _, row in qa_df.iterrows():
        question = row["question"]
        if question in already_done_questions:
            continue

        try:
            baseline_answer, baseline_passages = run_baseline(llm, search_tool, question)
            baseline_unsupported = judge_faithfulness(llm, baseline_answer, baseline_passages)
            time.sleep(sleep_seconds)

            crew_result = run_crew_fn(llm, search_tool, question, max_revisions=max_revisions)
            crew_unsupported = judge_faithfulness(llm, crew_result["answer"], crew_result["research_passages"])
            time.sleep(sleep_seconds)

            results.append({
                "question": question,
                "baseline_answer": baseline_answer,
                "baseline_unsupported_claims": baseline_unsupported,
                "crew_answer": crew_result["answer"],
                "crew_approved": crew_result["approved"],
                "crew_n_revisions": crew_result["n_revisions"],
                "crew_unsupported_claims": crew_unsupported,
                "failed": False,
            })
        except Exception as e:
            print(f"  Question failed ({type(e).__name__}: {str(e)[:150]}) -- logged, continuing.")
            results.append({
                "question": question,
                "baseline_answer": None,
                "baseline_unsupported_claims": None,
                "crew_answer": None,
                "crew_approved": False,
                "crew_n_revisions": 0,
                "crew_unsupported_claims": None,
                "failed": True,
            })

        if checkpoint_path and len(results) % checkpoint_every == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(results, f)
            print(f"  Checkpoint saved: {len(results)}/{len(qa_df)} questions done")

    if checkpoint_path:
        with open(checkpoint_path, "w") as f:
            json.dump(results, f)

    return results


def summarize(results):
    """Hallucination rate = fraction of judged answers with >=1 unsupported
    claim. Answers where the judge itself failed to parse are excluded
    from the rate (not counted as clean), and reported separately so a
    judge-parsing problem doesn't silently masquerade as "the model did
    great." Questions that failed outright (exception during the run, see
    run_comparison) are excluded from every stat here, not counted as
    "0 revisions, rejected" -- they never really ran.
    """
    ok_results = [r for r in results if not r.get("failed", False)]
    n_failed = len(results) - len(ok_results)

    def rate(key):
        judged = [r for r in ok_results if r[key] is not None]
        if not judged:
            return None
        return sum(1 for r in judged if len(r[key]) > 0) / len(judged)

    def judge_failure_count(key):
        return sum(1 for r in ok_results if r[key] is None)

    n = len(ok_results)
    return {
        "n_questions": len(results),
        "n_failed": n_failed,
        "baseline_hallucination_rate": rate("baseline_unsupported_claims"),
        "baseline_judge_failures": judge_failure_count("baseline_unsupported_claims"),
        "crew_hallucination_rate": rate("crew_unsupported_claims"),
        "crew_judge_failures": judge_failure_count("crew_unsupported_claims"),
        "crew_avg_revisions": (sum(r["crew_n_revisions"] for r in ok_results) / n) if n else None,
        "crew_approval_rate": (sum(1 for r in ok_results if r["crew_approved"]) / n) if n else None,
    }
