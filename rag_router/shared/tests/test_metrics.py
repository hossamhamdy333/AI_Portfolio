from metrics import reciprocal_rank, dcg_at_k, mean_mrr_ndcg, routing_accuracy, citation_accuracy


def test_reciprocal_rank_correct_is_first():
    assert reciprocal_rank(["a", "b", "c"], "a") == 1.0


def test_reciprocal_rank_not_found():
    assert reciprocal_rank(["a", "b", "c"], "z") == 0.0


def test_dcg_at_k_respects_k():
    assert dcg_at_k(["a", "b", "c"], "c", k=2) == 0.0


def test_mean_mrr_ndcg_aggregates():
    results_log = [
        {"ranked_ids": ["a", "b"], "correct_id": "a"},
        {"ranked_ids": ["b", "a"], "correct_id": "a"},
    ]
    scores = mean_mrr_ndcg(results_log, k=10)
    assert scores["mrr"] == 0.75


def test_routing_accuracy():
    results_log = [
        {"correct_domain": "sports", "routed_domain": "sports"},
        {"correct_domain": "tech", "routed_domain": "history"},
        {"correct_domain": "history", "routed_domain": "history"},
        {"correct_domain": "english_literature", "routed_domain": None},
    ]
    assert routing_accuracy(results_log) == 0.5


def test_citation_accuracy():
    results_log = [{"correct_cited": True}, {"correct_cited": False}]
    assert citation_accuracy(results_log) == 0.5
