"""RAGAS evaluation wrapper for the RAG chatbot.

MRR/NDCG scoring lives in shared/metrics.py, not here -- this file used to
redefine reciprocal_rank/dcg_at_k a second time (a third copy also lived
inline in 03_chunking.ipynb). One copy now, imported by whoever needs it.
"""


def run_ragas_eval(dataset, llm, embeddings):
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_recall

    results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_recall], llm=llm, embeddings=embeddings)
    return results.to_pandas()
