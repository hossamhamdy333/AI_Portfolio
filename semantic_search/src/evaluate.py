"""
Evaluation metrics for retrieval: MRR, NDCG, Recall@K
"""
import numpy as np


def mean_reciprocal_rank(rankings, correct_ids):
    """
    rankings: list of lists, each is a ranked list of doc ids for one query
    correct_ids: list of the correct doc id for each query
    """
    reciprocal_ranks = []
    for ranked_list, correct_id in zip(rankings, correct_ids):
        if correct_id in ranked_list:
            rank = ranked_list.index(correct_id) + 1   # rank is 1-indexed
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    return np.mean(reciprocal_ranks)


def recall_at_k(rankings, correct_ids, k):
    hits = 0
    for ranked_list, correct_id in zip(rankings, correct_ids):
        if correct_id in ranked_list[:k]:
            hits += 1
    return hits / len(rankings)


def ndcg_at_k(rankings, correct_ids, k):
    scores = []
    for ranked_list, correct_id in zip(rankings, correct_ids):
        if correct_id in ranked_list[:k]:
            rank = ranked_list[:k].index(correct_id) + 1
            dcg = 1.0 / np.log2(rank + 1)   # rank 1 -> highest score
        else:
            dcg = 0.0
        scores.append(dcg)
    return np.mean(scores)
