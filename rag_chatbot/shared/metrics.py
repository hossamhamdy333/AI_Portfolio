"""Shared scoring functions.

All three implementations (vanilla, langchain, llamaindex) call these same
functions so retrieval-quality numbers are computed identically everywhere.
RAGAS scoring itself is called directly from ragas in each impl's notebook
(no wrapper needed there), but MRR/NDCG/citation-accuracy live here since
those are hand-rolled and easy to accidentally compute differently.
"""

import numpy as np


def reciprocal_rank(ranked_ids, correct_id):
    """1 / rank of the first correct hit, 0 if not found."""
    for i, rid in enumerate(ranked_ids, start=1):
        if rid == correct_id:
            return 1.0 / i
    return 0.0


def dcg_at_k(ranked_ids, correct_id, k=10):
    for i, rid in enumerate(ranked_ids[:k], start=1):
        if rid == correct_id:
            return 1.0 / np.log2(i + 1)
    return 0.0


def ndcg_at_k(ranked_ids, correct_id, k=10):
    # ideal DCG here is 1.0 since there's exactly one correct id
    return dcg_at_k(ranked_ids, correct_id, k)


def mean_mrr_ndcg(eval_rows, k=10):
    """eval_rows: list of dicts with 'ranked_ids' and 'correct_id'."""
    mrrs = [reciprocal_rank(r["ranked_ids"], r["correct_id"]) for r in eval_rows]
    ndcgs = [ndcg_at_k(r["ranked_ids"], r["correct_id"], k) for r in eval_rows]
    return {"mrr": float(np.mean(mrrs)), "ndcg_at_k": float(np.mean(ndcgs))}


def citation_accuracy(results_log):
    """results_log: list of dicts with 'correct_cited' bool, as in impl_vanilla."""
    return float(np.mean([r["correct_cited"] for r in results_log]))


def routing_accuracy(results_log):
    """For impl_llamaindex only: did the router pick the right domain index?

    results_log: list of dicts with 'correct_domain' and 'routed_domain'.
    """
    return float(np.mean([r["correct_domain"] == r["routed_domain"] for r in results_log]))
