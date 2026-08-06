"""Orchestrates the retrieve -> generate RAG pipeline."""
from app import vectorstore, llm


def answer_question(question: str, top_k: int = None) -> dict:
    hits = vectorstore.query(question, top_k=top_k)
    answer = llm.generate_answer(question, hits)
    return {
        "question": question,
        "answer": answer,
<<<<<<< HEAD
        "sources": [{"source": h["source"], "score": round(h["score"], 3)} for h in hits],
=======
        "sources": [
            {"source": h["source"], "score": round(h["score"], 3), "text": h["text"]}
            for h in hits
        ],
>>>>>>> 1daa2d74e09f7db542620d4ab4861f9cf5e0dc25
    }
