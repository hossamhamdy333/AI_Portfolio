"""Run the eval set through the router and score it.

Combines three things per question:
  - routing accuracy (did the router pick the right domain index?)
  - citation accuracy (same check as impl_vanilla: is the correct article
    actually cited in the answer?)
  - the source_nodes returned, for MRR/NDCG scoring via shared/metrics.py

This intentionally mirrors impl_vanilla's eval-loop shape (sample the qa_df,
loop, log to MLflow, print a summary) so the two are easy to read side by
side when filling in COMPARISON.md.
"""

import time

import mlflow
import numpy as np

from router import route_and_query


def run_eval(router, domains, qa_df, mlflow_run_name="llamaindex_router_eval", sleep_seconds=1.0):
    """qa_df needs: question, answer, article_id, article_title, domain."""
    results_log = []

    with mlflow.start_run(run_name=mlflow_run_name):
        for _, row in qa_df.iterrows():
            answer_text, routed_domain, source_nodes = route_and_query(router, domains, row["question"])

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
            time.sleep(sleep_seconds)

        routing_acc = np.mean([r["correct_domain"] == r["routed_domain"] for r in results_log])
        citation_acc = np.mean([r["correct_cited"] for r in results_log])

        mlflow.log_metric("routing_accuracy", routing_acc)
        mlflow.log_metric("citation_accuracy", citation_acc)

    return results_log, {"routing_accuracy": routing_acc, "citation_accuracy": citation_acc}
