"""RAGAS evaluation and retrieval scoring for the RAG chatbot."""

import numpy as np


def reciprocal_rank(retrieved_ids, correct_id):
    if correct_id in retrieved_ids:
        return 1.0 / (retrieved_ids.index(correct_id) + 1)
    return 0.0


def dcg_at_k(retrieved_ids, correct_id, k):
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid == correct_id:
            return 1.0 / np.log2(i + 2)
    return 0.0


def run_ragas_eval(dataset, llm, embeddings):
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_recall

    results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_recall], llm=llm, embeddings=embeddings)
    return results.to_pandas()
