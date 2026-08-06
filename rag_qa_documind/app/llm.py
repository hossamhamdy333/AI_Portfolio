"""Thin wrapper around the Gemini API (Interactions API) for the generation
step of RAG. Google requires the Interactions API -- not the older
generateContent method -- for current-generation models like gemini-3.6-flash;
using generateContent with these models returns a confusing
'unexpected model name format' error instead of a clear deprecation notice."""
from google import genai

from app.config import settings

_client = None


def get_client() -> "genai.Client":
    global _client
    if _client is None:
        if not settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to your .env file. "
                "Get a free key at https://aistudio.google.com/apikey"
            )
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


SYSTEM_PROMPT = """You are DocuMind, a precise Q&A assistant that answers ONLY \
using the provided context excerpts from the user's documents.

Rules:
- If the answer is not contained in the context, say you don't have enough \
information in the provided documents. Do not use outside knowledge.
- Cite which source(s) you used, by filename, at the end of your answer.
- Be concise and direct.
"""


def generate_answer(question: str, context_chunks: list) -> str:
    if not context_chunks:
        return (
            "I don't have any indexed documents to answer from yet. "
            "Please ingest some documents first."
        )

    context_block = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
    )

    user_message = f"""Context excerpts:
{context_block}

Question: {question}

Answer using only the context above."""

    client = get_client()
    interaction = client.interactions.create(
        model=settings.gemini_model,
        input=user_message,
        system_instruction=SYSTEM_PROMPT,
    )
    return interaction.output_text
