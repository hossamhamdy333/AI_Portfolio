"""Orchestrates the retrieve -> generate RAG pipeline."""
from app import vectorstore, llm


def answer_question(question: str, top_k: int = None) -> dict:
    hits = vectorstore.query(question, top_k=top_k)
    answer = llm.generate_answer(question, hits)
    return {
        "question": question,
        "answer": answer,
        "sources": [
            {"source": h["source"], "score": round(h["score"], 3), "text": h["text"]}
            for h in hits
        ],
    }
