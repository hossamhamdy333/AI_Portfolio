"""Retrieval scoring for rag_router.

A near-identical module exists at ../rag_chatbot/shared/metrics.py -- not
imported from here on purpose. rag_router is a standalone project (own
corpus, own task, own config); reaching across into a sibling project's
internals for a ~20-line function would be tighter coupling than the
duplication costs. See each project's own README for the reasoning.
"""

import numpy as np


def reciprocal_rank(ranked_ids, correct_id):
    if correct_id in ranked_ids:
        return 1.0 / (ranked_ids.index(correct_id) + 1)
    return 0.0


def dcg_at_k(ranked_ids, correct_id, k):
    for i, rid in enumerate(ranked_ids[:k]):
        if rid == correct_id:
            return 1.0 / np.log2(i + 2)
    return 0.0


def mean_mrr_ndcg(results_log, k=10):
    """results_log: list of dicts, each needs 'ranked_ids' and 'correct_id'."""
    mrr_scores = [reciprocal_rank(r["ranked_ids"], r["correct_id"]) for r in results_log]
    ndcg_scores = [dcg_at_k(r["ranked_ids"], r["correct_id"], k) for r in results_log]
    return {"mrr": float(np.mean(mrr_scores)), "ndcg_at_k": float(np.mean(ndcg_scores))}


def routing_accuracy(results_log):
    """results_log: list of dicts, each needs 'correct_domain' and 'routed_domain'."""
    return float(np.mean([r["correct_domain"] == r["routed_domain"] for r in results_log]))


def citation_accuracy(results_log):
    """results_log: list of dicts, each needs 'correct_cited' (bool)."""
    return float(np.mean([r["correct_cited"] for r in results_log]))
