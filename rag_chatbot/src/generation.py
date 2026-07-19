"""RAG generation with citation injection.

Cost/tracking/retry logic adapted from llm_api_integration/src/client.py
and tracking.py -- copied here, not imported, per portfolio standalone-project rule.
Uses google-genai (google.generativeai is deprecated).
"""

import re
import time
import logging

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

GENERATION_PROMPT_TEMPLATE = """You are answering a question using only the sources below.
Tag every claim with the source number it came from, like [1] or [2].
If the sources do not contain the answer, say so -- do not guess.

Sources:
{sources_block}

Question: {question}

Answer (in Arabic, with [N] citation tags):"""


def build_sources_block(chunks):
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append("[" + str(i) + "] " + c["chunk_text"])
    sep = chr(10) + chr(10)
    return sep.join(lines)


def call_gemini(client, model_name, gen_config, prompt, max_attempts=3, backoff_seconds=2):
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return client.models.generate_content(model=model_name, contents=prompt, config=gen_config)
        except Exception as e:
            last_error = e
            wait = backoff_seconds * attempt
            match = re.search(r"retry in ([\d.]+)s", str(e))
            if match:
                wait = float(match.group(1)) + 1
            logger.warning("Gemini call failed (attempt " + str(attempt) + "/" + str(max_attempts) + "): " + str(e))
            if attempt < max_attempts:
                time.sleep(wait)
    raise RuntimeError("Gemini call failed after " + str(max_attempts) + " attempts") from last_error


def generate_answer(client, question, chunks, model_name, temperature, max_output_tokens):
    sources_block = build_sources_block(chunks)
    prompt = GENERATION_PROMPT_TEMPLATE.format(sources_block=sources_block, question=question)
    gen_config = types.GenerateContentConfig(temperature=temperature, max_output_tokens=max_output_tokens)
    response = call_gemini(client, model_name, gen_config, prompt)

    cited_ids_used = set(int(n) for n in re.findall(r"\[(\d+)\]", response.text))
    citations = [
        {"source_num": i, "article_id": c["article_id"], "chunk_id": c["chunk_id"]}
        for i, c in enumerate(chunks, start=1) if i in cited_ids_used
    ]
    return response.text, citations, response


def compute_cost_usd(prompt_tokens, response_tokens, input_rate, output_rate):
    return (prompt_tokens / 1000000) * input_rate + (response_tokens / 1000000) * output_rate


def extract_token_counts(response):
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return 0, 0
    return usage.prompt_token_count, usage.candidates_token_count
