"""
RAG evaluation with Ragas — free, open source, uses Groq as the judge LLM
(no OpenAI key needed, which is Ragas's default judge).

Run: python eval/evaluate_rag.py
Edit SAMPLE_QA below with real questions once you've indexed real docs.
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate, EvaluationDataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from ingestion.chunking import get_qdrant, search

QDRANT_PATH = os.getenv("QDRANT_PATH", "./data/qdrant_db")

# Replace with real (question, ground_truth) pairs relevant to your indexed docs.
SAMPLE_QA = [
    {"question": "What is the main topic of the indexed documents?",
     "ground_truth": "Replace with the correct answer for your own docs."},
]


def build_dataset(qdrant_client, qa_pairs):
    rows = []
    for qa in qa_pairs:
        hits = search(qdrant_client, qa["question"], k=4)
        contexts = [h["text"] for h in hits]
        rows.append({
            "question": qa["question"],
            "contexts": contexts,
            "answer": " ".join(contexts)[:500],  # swap in your agent's real answer if desired
            "ground_truth": qa["ground_truth"],
        })
    return rows


def main():
    judge_llm = LangchainLLMWrapper(ChatGroq(model="llama-3.3-70b-versatile", temperature=0))
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    qdrant_client = get_qdrant(QDRANT_PATH)
    rows = build_dataset(qdrant_client, SAMPLE_QA)
    dataset = EvaluationDataset.from_list(rows)

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )
    print(result)
    result.to_pandas().to_csv("eval/ragas_results.csv", index=False)
    print("Saved eval/ragas_results.csv")


if __name__ == "__main__":
    main()
