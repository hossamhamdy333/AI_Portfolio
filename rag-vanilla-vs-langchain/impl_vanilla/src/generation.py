"""RAG generation with citation injection.

Retry/cost/token-tracking logic lives in shared/llm_client.py -- this file
only owns the prompt template, source formatting, and citation parsing
specific to this pipeline.
"""

import re

from google.genai import types

from shared.llm_client import call_gemini

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


def generate_answer(client, question, chunks, model_name, temperature, max_output_tokens):
    sources_block = build_sources_block(chunks)
    prompt = GENERATION_PROMPT_TEMPLATE.format(sources_block=sources_block, question=question)
    gen_config = types.GenerateContentConfig(temperature=temperature, max_output_tokens=max_output_tokens)
    response = call_gemini(client, model_name, gen_config, prompt)

    # response.text can be None -- a normal return, not an exception --
    # when Gemini's finish_reason isn't one the installed google-genai SDK
    # recognizes. Treat it as an empty answer with no citations rather
    # than crashing re.findall() on None.
    answer_text = response.text or ""
    cited_ids_used = set(int(n) for n in re.findall(r"\[(\d+)\]", answer_text))
    citations = [
        {"source_num": i, "article_id": c["article_id"], "chunk_id": c["chunk_id"]}
        for i, c in enumerate(chunks, start=1) if i in cited_ids_used
    ]
    return answer_text, citations, response
