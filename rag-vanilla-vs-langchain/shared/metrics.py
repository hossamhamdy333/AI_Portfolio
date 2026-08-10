"""Retrieval scoring shared by every notebook/eval script in this repo.

This used to be implemented three times (impl_vanilla/src/evaluate.py,
this file, and again inline in impl_vanilla's 03_chunking.ipynb) -- all
three copies did the same MRR/NDCG math. This is now the one copy;
everything else imports it.
"""

import numpy as np


def reciprocal_rank(ranked_ids, correct_id):
    """1 / rank of the first correct hit, 0 if it never shows up."""
    if correct_id in ranked_ids:
        return 1.0 / (ranked_ids.index(correct_id) + 1)
    return 0.0


def dcg_at_k(ranked_ids, correct_id, k):
    """Binary-relevance DCG: 1/log2(rank+1) for the correct hit, 0 otherwise.
    There's exactly one relevant document per question in this eval set, so
    IDCG is always 1.0 and this doubles as NDCG@k directly.
    """
    for i, rid in enumerate(ranked_ids[:k]):
        if rid == correct_id:
            return 1.0 / np.log2(i + 2)
    return 0.0


def mean_mrr_ndcg(results_log, k=10):
    """results_log: list of dicts, each needs 'ranked_ids' and 'correct_id'.

    Returns {"mrr": ..., "ndcg_at_k": ...} averaged over every row.
    """
    mrr_scores = [reciprocal_rank(r["ranked_ids"], r["correct_id"]) for r in results_log]
    ndcg_scores = [dcg_at_k(r["ranked_ids"], r["correct_id"], k) for r in results_log]
    return {"mrr": float(np.mean(mrr_scores)), "ndcg_at_k": float(np.mean(ndcg_scores))}


def citation_accuracy(results_log):
    """results_log: list of dicts, each needs 'correct_cited' (bool)."""
    return float(np.mean([r["correct_cited"] for r in results_log]))
