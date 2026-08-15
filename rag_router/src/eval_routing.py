"""Run the eval set through a router and score it.

Combines three things per question:
  - routing accuracy (did the router pick the right domain?)
  - citation accuracy (is the correct article actually cited in the answer?)
  - the source_nodes returned, for MRR/NDCG scoring via shared/metrics.py

Rate-limit handling: each question triggers ~1-2 Gemini calls depending on
selector (embedding selector: 1 call, for generation only; LLM selector: 2,
routing + generation). Free-tier gemini-3.1-flash-lite caps at 15
requests/min, so configs/config.yaml's eval.sleep_seconds=12 stays under
that with margin.
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
        return float(match.group(1)) + 2
    return default


def _is_selector_parsing_bug(error):
    """LlamaIndex's selector-result parsing (Answer/SingleSelection via
    dataclasses_json) occasionally decodes a response with choice=None,
    which crashes downstream with exactly this TypeError when the choice
    gets converted to a 0-indexed domain. This is deterministic for a given
    question -- retrying the identical call 5 times produces the identical
    crash 5 times (confirmed against this project's own past eval runs,
    where every attempt failed with the same error before giving up) -- so
    it needs different handling than a transient rate limit does.
    """
    return isinstance(error, TypeError) and "NoneType" in str(error) and "int" in str(error)


def _route_with_retry(router, domains, question, max_attempts=5):
    """Rate limits (429) get their recommended wait time and a real retry,
    since those are genuinely transient. The selector-parsing TypeError
    (see _is_selector_parsing_bug) short-circuits after the first
    occurrence instead -- retrying it can't succeed, so spending 4 more
    attempts on it only wastes ~10x the wall-clock time for an outcome
    that's already decided. Any other failure gets one short fixed-backoff
    retry pass, same as before.
    """
    from router import route_and_query

    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return route_and_query(router, domains, question)
        except Exception as e:
            last_error = e
            if _is_selector_parsing_bug(e):
                print(f"  Selector parsing bug (LlamaIndex's choice=None decode issue) -- "
                      f"not retrying, this is deterministic for this question.")
                break
            elif "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = _extract_retry_seconds(e)
                print(f"  Rate limited (attempt {attempt}/{max_attempts}), waiting {wait:.0f}s...")
                time.sleep(wait)
            else:
                wait = 5
                print(f"  Failed (attempt {attempt}/{max_attempts}): {type(e).__name__}: {str(e)[:150]}")
                time.sleep(wait)

    print(f"  Giving up on this question, logging as failed: {type(last_error).__name__}")
    return f"[FAILED: {type(last_error).__name__}]", None, []


def run_eval(router, domains, qa_df, selector_name, mlflow_run_name=None,
             sleep_seconds=12, checkpoint_path=None, checkpoint_every=10):
    """qa_df needs: question, answer, article_id, article_title, domain.

    selector_name: "llm" or "embedding" -- used to build a unique
    checkpoint filename and MLflow run name, so two different selectors
    (or two separate attempts at the same selector) never silently share
    or overwrite each other's checkpoint. A real bug in an earlier version
    of this project had two separate Colab sessions resume from the same
    un-versioned checkpoint file at two different points, producing two
    different "final" scores for what was supposed to be one run -- see
    COMPARISON.md.

    checkpoint_path: if given, results are saved here as JSON every
    `checkpoint_every` questions, and any existing checkpoint is loaded
    first so a rerun after a crash resumes instead of restarting. Delete
    the checkpoint file yourself if you intend to start a genuinely fresh
    run rather than resume.
    """
    mlflow_run_name = mlflow_run_name or f"router_eval_{selector_name}"
    results_log = []
    already_done_questions = set()

    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            results_log = json.load(f)
        already_done_questions = {r["question"] for r in results_log}
        print(f"Resuming '{selector_name}' eval from checkpoint: {len(results_log)} questions already done")

    with mlflow.start_run(run_name=mlflow_run_name):
        mlflow.log_param("selector", selector_name)

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
