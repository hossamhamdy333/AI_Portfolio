import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_retrieval_eval import COLLECTION, ROOT, build_index, failed_metrics

from shared.eval_set import sample_eval_set
from shared.llm_client import get_client
from src.generation import generate_answer
from src.retrieval import rerank_chunks, search_chunks

TOLERANCE = 0.05
METRICS = ["faithfulness", "answer_relevancy", "context_recall"]


def make_rows(cfg, questions, qdrant, embedder, reranker, gemini):
    rows = []
    for row in questions.itertuples():
        found = search_chunks(row.question, qdrant, COLLECTION, embedder, top_k=cfg["rag"]["top_k_retrieve"])
        top = rerank_chunks(row.question, found, reranker, top_k=cfg["rag"]["top_k_rerank"])
        try:
            answer, _, _ = generate_answer(
                gemini, row.question, top, cfg["rag"]["llm_model"],
                cfg["rag"]["temperature"], cfg["rag"]["max_output_tokens"],
            )
        except Exception as e:
            print("generation failed, skipping question:", str(e)[:120])
            continue
        rows.append({
            "user_input": row.question,
            "response": answer,
            "retrieved_contexts": [c["chunk_text"] for c in top],
            "reference": row.answer,
        })
        print("answered", len(rows))
        time.sleep(5)
    return rows


def score_rows(cfg, rows):
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerRelevancy, ContextRecall, Faithfulness

    key = os.environ["GEMINI_API_KEY"]
    llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model=cfg["rag"]["llm_model"], google_api_key=key))
    emb = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=key))
    metrics = [Faithfulness(), AnswerRelevancy(), ContextRecall()]

    results = []
    for i, row in enumerate(rows):
        for attempt in (1, 2):
            try:
                out = evaluate(EvaluationDataset.from_list([row]), metrics=metrics, llm=llm, embeddings=emb)
                results.append(out.to_pandas().iloc[0].to_dict())
                print("scored", i + 1, "of", len(rows))
                break
            except Exception as e:
                print("row", i + 1, "attempt", attempt, "failed:", str(e)[:150])
                time.sleep(60)
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml", encoding="utf-8"))
    chunks = pd.read_parquet(HERE / "chunks.parquet")
    questions = pd.read_parquet(HERE / "questions.parquet")
    questions = sample_eval_set(questions, args.n, cfg["data"]["random_seed"])

    from sentence_transformers import CrossEncoder, SentenceTransformer
    embedder = SentenceTransformer(cfg["retrieval"]["model_name"])
    reranker = CrossEncoder(cfg["reranker"]["model_name"])
    qdrant = build_index(chunks, embedder)

    rows = make_rows(cfg, questions, qdrant, embedder, reranker, get_client())
    df = score_rows(cfg, rows)
    df.to_csv(HERE / "ragas_last.csv", index=False)

    scores = {m: round(float(df[m].mean()), 4) for m in METRICS}
    print("questions answered:", len(rows), "scored:", len(df))
    print("scores:", json.dumps(scores))

    baseline_path = HERE / "ragas_baseline.json"
    if args.write_baseline:
        baseline_path.write_text(json.dumps(scores, indent=2))
        print("baseline written")
        return
    if not baseline_path.exists():
        print("No eval/ragas_baseline.json yet. Copy the scores line above into it.")
        sys.exit(1)

    baseline = json.loads(baseline_path.read_text())
    for name, base in baseline.items():
        print(f"{name}: now {scores[name]} | baseline {base}")
    failed = failed_metrics(scores, baseline, TOLERANCE)
    if failed:
        print("RAGAS GATE FAILED, dropped more than", TOLERANCE, "on:", ", ".join(failed))
        sys.exit(1)
    print("RAGAS GATE PASSED")


if __name__ == "__main__":
    main()
