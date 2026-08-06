"""Run the eval set through the router and score it.

Combines three things per question:
  - routing accuracy (did the router pick the right domain index?)
  - citation accuracy (same check as impl_vanilla: is the correct article
    actually cited in the answer?)
  - the source_nodes returned, for MRR/NDCG scoring via shared/metrics.py

This intentionally mirrors impl_vanilla's eval-loop shape (sample the qa_df,
loop, log to MLflow, print a summary) so the two are easy to read side by
side when filling in COMPARISON.md.

Rate-limit handling: each question triggers ~2 Gemini calls (router selector
+ answer generation). Free-tier gemini-3.1-flash-lite caps at 15 requests/min,
so the default pacing here (8s between questions) stays under that. On a 429
(quota exceeded) that survives the client's own internal retry, this waits
and retries rather than crashing -- and results are checkpointed to disk
periodically, so a crash or interrupted session doesn't lose prior progress;
rerunning picks up where it left off instead of starting from zero.
"""

import json
import os
import re
import time

import mlflow
import numpy as np


def _extract_retry_seconds(error, default=60):
    """Gemini's error message includes 'Please retry in Xs' -- parse it out
    so we wait the actual recommended time instead of guessing."""
    match = re.search(r"retry in ([\d.]+)s", str(error))
    if match:
        return float(match.group(1)) + 2  # small buffer
    return default


def _route_with_retry(router, domains, question, max_attempts=5):
    """Retries on any failure -- rate limits (429) get their recommended
    wait time; other transient failures (e.g. LlamaIndex's selector
    occasionally returning a malformed response with no valid 'choice'
    field) get a short fixed backoff. Never raises: if all attempts fail,
    returns a placeholder result so one bad question can't crash the whole
    run and lose everything already checkpointed.
    """
    from router import route_and_query

    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return route_and_query(router, domains, question)
        except Exception as e:
            last_error = e
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = _extract_retry_seconds(e)
                print(f"  Rate limited (attempt {attempt}/{max_attempts}), waiting {wait:.0f}s...")
            else:
                wait = 5
                print(f"  Failed (attempt {attempt}/{max_attempts}): {type(e).__name__}: {str(e)[:150]}")
            time.sleep(wait)

    print(f"  Giving up on this question after {max_attempts} attempts, logging as failed.")
    return f"[FAILED: {type(last_error).__name__}]", None, []


def run_eval(router, domains, qa_df, mlflow_run_name="llamaindex_router_eval",
             sleep_seconds=8.0, checkpoint_path=None, checkpoint_every=10):
    """qa_df needs: question, answer, article_id, article_title, domain.

    checkpoint_path: if given, results are saved here as JSON every
    `checkpoint_every` questions, and any existing checkpoint is loaded
    first so a rerun after a crash resumes instead of restarting.
    """
    results_log = []
    already_done_questions = set()

    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            results_log = json.load(f)
        already_done_questions = {r["question"] for r in results_log}
        print(f"Resuming from checkpoint: {len(results_log)} questions already done")

    with mlflow.start_run(run_name=mlflow_run_name):
        for i, (_, row) in enumerate(qa_df.iterrows()):
            if row["question"] in already_done_questions:
                continue

            answer_text, routed_domain, source_nodes = _route_with_retry(router, domains, row["question"])

            cited_article_ids = {node.node.metadata.get("title") for node in source_nodes} if source_nodes else set()
            correct_cited = row["article_title"] in cited_article_ids

            ranked_ids = [node.node.metadata.get("title") for node in source_nodes] if source_nodes else []

            results_log.append({
                "question": row["question"],
                "correct_domain": row["domain"],
                "routed_domain": routed_domain,
                "correct_article": row["article_title"],
                "cited_articles": list(cited_article_ids),
                "correct_cited": correct_cited,
                "ranked_ids": ranked_ids,
                "correct_id": row["article_title"],
            })

            if checkpoint_path and len(results_log) % checkpoint_every == 0:
                with open(checkpoint_path, "w") as f:
                    json.dump(results_log, f)
                print(f"  Checkpoint saved: {len(results_log)} questions done")

            time.sleep(sleep_seconds)

        if checkpoint_path:
            with open(checkpoint_path, "w") as f:
                json.dump(results_log, f)

        routing_acc = np.mean([r["correct_domain"] == r["routed_domain"] for r in results_log])
        citation_acc = np.mean([r["correct_cited"] for r in results_log])

        mlflow.log_metric("routing_accuracy", routing_acc)
        mlflow.log_metric("citation_accuracy", citation_acc)

    return results_log, {"routing_accuracy": routing_acc, "citation_accuracy": citation_acc}
