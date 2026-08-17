"""Synthetic Q&A generation for the 4-domain Wikipedia corpus.

Retry/cost/token-tracking logic lives in shared/llm_client.py -- this file
only owns the prompt template and response parsing specific to this
corpus, and tags every row with its source domain so
notebooks/04_evaluation.ipynb can check the router's pick against ground
truth.
"""

import time

from llm_client import (
    call_gemini,
    compute_cost_usd,
    extract_token_counts,
    get_client,
    log_usage,
    strip_markdown_fences,
)
from llm_client import build_generation_config
import json


def build_model(model_name: str, temperature: float, max_output_tokens: int, json_mode: bool = False):
    return model_name, build_generation_config(temperature, max_output_tokens, json_mode)


QA_PROMPT_TEMPLATE = """You are generating evaluation questions for a retrieval system.

Given this Wikipedia article, write {n_questions} questions that this article directly answers.
Each question must be answerable using only the information in the article, and must be specific
enough that it couldn't plausibly be answered by an article from a different topic domain.
For each question, also provide a short factual answer (1-2 sentences) taken directly from the article.

Respond ONLY with valid JSON in this exact format, no other text:
{{"qa_pairs": [{{"question": "question 1", "answer": "short answer 1"}}, {{"question": "question 2", "answer": "short answer 2"}}]}}

Article:
{article_text}
"""


def generate_questions(model, article_text: str, n_questions: int = 2):
    # get_client() is lazy -- only connects on first real use, not at
    # import time, so importing this module for testing never requires a
    # live API key (same fix applied to rag-vanilla-vs-langchain's
    # impl_vanilla/src/qa_generation.py after the original version crashed
    # on import without one).
    model_name, gen_config = model
    prompt = QA_PROMPT_TEMPLATE.format(n_questions=n_questions, article_text=article_text[:2000])
    response = call_gemini(get_client(), model_name, gen_config, prompt)
    raw_text = strip_markdown_fences(response.text)
    try:
        parsed = json.loads(raw_text)
        return parsed.get("qa_pairs", []), response
    except json.JSONDecodeError:
        return [], response


def generate_qa_dataset(model, domain_dfs: dict, config: dict, sleep_seconds: float = 4.5):
    """domain_dfs: dict of domain -> DataFrame (columns: article_id, title, text).

    Generates questions per domain and tags every row with its source
    domain, so eval_routing.py can check the router's pick against ground
    truth. Caller must run shared.tracking.init_tracking(...) first for
    MLflow logging.
    """
    qa_pairs = []
    failed = []
    total_cost = 0.0

    for domain, df in domain_dfs.items():
        for _, row in df.iterrows():
            try:
                questions, response = generate_questions(
                    model, row["text"], n_questions=config["synthetic_qa"]["questions_per_article"]
                )
                p_tok, r_tok = extract_token_counts(response)
                total_cost += log_usage(
                    p_tok, r_tok, config["synthetic_qa"]["model"],
                    config["cost"]["input_rate_per_million"], config["cost"]["output_rate_per_million"]
                )
                for qa in questions:
                    qa_pairs.append({
                        "question": qa.get("question", ""),
                        "answer": qa.get("answer", ""),
                        "article_id": row["article_id"],
                        "article_title": row["title"],
                        "domain": domain,
                    })
            except Exception as e:
                failed.append({"article_id": row["article_id"], "domain": domain, "error": str(e)})

            time.sleep(sleep_seconds)

    return qa_pairs, failed, total_cost
