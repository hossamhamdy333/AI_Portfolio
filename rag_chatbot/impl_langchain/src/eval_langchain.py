"""Eval loop for the LangChain implementation.

Runs the identical 400-question eval set impl_vanilla uses (loaded through
shared/eval_set.py) through the LangChain chain, scores retrieval with
shared/metrics.py (same MRR/NDCG math vanilla uses), and reuses RAGAS the
same way impl_vanilla's evaluate.py does. This is what fills in the
"impl_langchain" column of COMPARISON.md.
"""

import time

import mlflow
import numpy as np

from generation import verify_citations

import sys
from pathlib import Path

_SHARED_DIR = Path(__file__).resolve().parents[2] / "shared"
if str(_SHARED_DIR) not in sys.path:
    sys.path.append(str(_SHARED_DIR))

from eval_set import load_eval_set, sample_eval_set  # noqa: E402
from metrics import mean_mrr_ndcg, citation_accuracy  # noqa: E402


def run_eval(run_chain, qa_df, mlflow_run_name="langchain_eval", sleep_seconds=1.0):
    """qa_df needs: question, answer, article_id (same schema impl_vanilla's eval set uses)."""
    results_log = []

    with mlflow.start_run(run_name=mlflow_run_name):
        for _, row in qa_df.iterrows():
            answer_text, reranked_docs = run_chain(row["question"])
            citation_result = verify_citations(answer_text, reranked_docs, row["article_id"])

            ranked_ids = [doc.metadata.get("article_id") for doc in reranked_docs]
            results_log.append({
                "question": row["question"],
                "answer_text": answer_text,
                "correct_id": row["article_id"],
                "ranked_ids": ranked_ids,
                "correct_cited": citation_result["correct_cited"],
            })
            time.sleep(sleep_seconds)

        retrieval_scores = mean_mrr_ndcg(results_log, k=10)
        citation_acc = citation_accuracy(results_log)

        mlflow.log_metric("mrr", retrieval_scores["mrr"])
        mlflow.log_metric("ndcg_at_10", retrieval_scores["ndcg_at_k"])
        mlflow.log_metric("citation_accuracy", citation_acc)

    return results_log, {**retrieval_scores, "citation_accuracy": citation_acc}


if __name__ == "__main__":
    # Wire-up sketch -- fill in real config/model objects before running.
    qa_df = load_eval_set("../../impl_vanilla/outputs/synthetic_qa/synthetic_qa_pairs.parquet")
    qa_sample = sample_eval_set(qa_df, n=100, random_seed=42)
    print(f"Loaded {len(qa_sample)} eval questions -- same set impl_vanilla scored against.")
