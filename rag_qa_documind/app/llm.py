from google import genai
from google.genai import types

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
    # Uses the standard generate_content endpoint rather than the newer
    # Interactions API (client.interactions.create). As of mid-2026, Google
    # is rolling out a new "AQ." API key format, and the Interactions API
    # has an active, widely-reported bug rejecting AQ-format keys with
    # 401 ACCESS_TOKEN_TYPE_UNSUPPORTED -- see
    # https://discuss.ai.google.dev/t/not-able-to-use-api-key-starting-with-aq/174115
    # generate_content is the long-established endpoint and doesn't have
    # this issue, so it's the safer choice for a free-tier API key today.
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=user_message,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    return response.text
