"""
Faithfulness evaluator: checks whether a generated response is actually
supported by the retrieved context, or whether it hallucinated.

Uses the unified `google-genai` SDK. Note: the older `google-generativeai`
package (import google.generativeai as genai) is deprecated -- Google's own
migration notice says post-mid-2026 SDK releases don't support new Gemini
models at all, so this intentionally uses the current package.

GEMINI_API_KEY is one shared key for the whole app (this is a portfolio
project, not a product -- asking every visitor to bring their own API key
would be a worse experience for zero real benefit). To keep that shared
key from being run up by traffic, every call is metered against
GEMINI_DAILY_LIMIT via the GeminiUsage table -- once the day's calls are
used, the check is skipped for the rest of the day rather than erroring
the whole chat request.
"""

import json
import logging
import os
from datetime import date

from google import genai
from google.genai import types
from sqlalchemy.orm import Session

from src.config import settings
from src.models import GeminiUsage

logging.basicConfig(level=logging.INFO)

MODEL_NAME = "gemini-3.1-flash-lite"


def _under_daily_limit(db: Session) -> bool:
    """Returns True (and reserves a slot) if today's Gemini call count is
    still under the limit; False if the daily budget is already used up."""
    today = str(date.today())
    row = db.query(GeminiUsage).filter(GeminiUsage.date == today).first()
    if row is None:
        row = GeminiUsage(date=today, count=0)
        db.add(row)
    if row.count >= settings.GEMINI_DAILY_LIMIT:
        return False
    row.count += 1
    db.commit()
    return True


def evaluate_faithfulness(query: str, context: str, ai_response: str, db: Session):
    if not _under_daily_limit(db):
        logging.info("Gemini daily limit (%d) reached -- skipping faithfulness check.", settings.GEMINI_DAILY_LIMIT)
        return None

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logging.error("Missing GEMINI_API_KEY environment variable. Get a free one at https://aistudio.google.com/")
        return None

    client = genai.Client(api_key=api_key)

    prompt = f"""Evaluate if the 'Assistant Response' is fully supported by the 'Context' (no hallucinations).
Context: {context}
User Query: {query}
Assistant Response: {ai_response}

Output strict JSON with EXACTLY these two keys:
1. "is_faithful": boolean (true if supported, false if hallucinated)
2. "reason": "string (short explanation)"
"""

    try:
        logging.info("Sending evaluation request to Gemini...")
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        return json.loads(response.text)
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        return None
