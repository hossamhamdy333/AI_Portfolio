"""Synthetic Q&A generation for the XLSum Arabic eval set.

Retry/cost/token-tracking logic lives in shared/llm_client.py -- this file
only owns the prompt template and response parsing specific to this corpus.
"""

import json

from shared.llm_client import call_gemini, get_client, strip_markdown_fences

QA_PROMPT_TEMPLATE = """You are generating evaluation questions for a retrieval system.

Given this Arabic news article, write {n_questions} questions that this article directly answers.
Each question must be answerable using only the information in the article.
For each question, also provide a short factual answer (1-2 sentences) taken directly from the article.

Respond ONLY with valid JSON in this exact format, no other text:
{{"qa_pairs": [{{"question": "question 1 in Arabic", "answer": "short answer 1 in Arabic"}}, {{"question": "question 2 in Arabic", "answer": "short answer 2 in Arabic"}}]}}

Article:
{article_text}
"""


def generate_questions(model, article_text: str, n_questions: int = 2):
    # get_client() is lazy -- only actually connects when this function is
    # called, not at import time. The old version called genai.Client() as
    # a module-level statement, which meant just *importing* this file (as
    # tests/test_qa_generation.py does) crashed without a live API key.
    model_name, gen_config = model
    prompt = QA_PROMPT_TEMPLATE.format(n_questions=n_questions, article_text=article_text[:2000])
    response = call_gemini(get_client(), model_name, gen_config, prompt)
    raw_text = strip_markdown_fences(response.text)
    try:
        parsed = json.loads(raw_text)
        return parsed.get("qa_pairs", []), response
    except json.JSONDecodeError:
        return [], response


def generate_qa_dataset(model, df, config: dict, sleep_seconds: float = 4.5):
    """Generate synthetic Q&A pairs for every row in df (needs id, title, article).
    Caller must run shared.llm_client.init_tracking(...) first for MLflow logging.
    """
    import time
    from shared.llm_client import extract_token_counts, log_usage

    qa_pairs = []
    failed = []
    total_cost = 0.0

    for _, row in df.iterrows():
        try:
            questions, response = generate_questions(
                model, row["article"], n_questions=config["synthetic_qa"]["questions_per_article"]
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
                    "article_id": row["id"],
                    "article_title": row["title"]
                })
        except Exception as e:
            failed.append({"article_id": row["id"], "error": str(e)})

        time.sleep(sleep_seconds)

    return qa_pairs, failed, total_cost
