"""Tests for shared/metrics.py -- MRR/NDCG against known values."""

from shared.metrics import reciprocal_rank, dcg_at_k


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
