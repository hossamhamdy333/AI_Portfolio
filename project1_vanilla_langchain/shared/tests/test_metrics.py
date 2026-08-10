"""Tests for shared/metrics.py -- previously this logic existed in three
places and none of them had tests. One copy, one test file now.
"""

from metrics import reciprocal_rank, dcg_at_k, mean_mrr_ndcg, citation_accuracy


def test_reciprocal_rank_correct_is_first():
    assert reciprocal_rank(["a", "b", "c"], "a") == 1.0


def test_reciprocal_rank_correct_is_second():
    assert reciprocal_rank(["a", "b", "c"], "b") == 0.5


def test_reciprocal_rank_not_found():
    assert reciprocal_rank(["a", "b", "c"], "z") == 0.0


def test_dcg_at_k_correct_is_first():
    assert dcg_at_k(["a", "b", "c"], "a", k=10) == 1.0


def test_dcg_at_k_not_in_top_k():
    assert dcg_at_k(["a", "b", "c"], "z", k=2) == 0.0


def test_dcg_at_k_respects_k():
    # correct id is at index 2, but k=2 only looks at the first 2 slots
    assert dcg_at_k(["a", "b", "c"], "c", k=2) == 0.0


def test_mean_mrr_ndcg_aggregates_correctly():
    results_log = [
        {"ranked_ids": ["a", "b"], "correct_id": "a"},   # mrr=1.0
        {"ranked_ids": ["b", "a"], "correct_id": "a"},   # mrr=0.5
    ]
    scores = mean_mrr_ndcg(results_log, k=10)
    assert scores["mrr"] == 0.75
    assert 0.0 < scores["ndcg_at_k"] <= 1.0


def test_citation_accuracy():
    results_log = [{"correct_cited": True}, {"correct_cited": True}, {"correct_cited": False}, {"correct_cited": True}]
    assert citation_accuracy(results_log) == 0.75
