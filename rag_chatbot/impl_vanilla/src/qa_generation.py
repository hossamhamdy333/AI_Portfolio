"""Synthetic Q&A generation eval set.

Cost/tracking/retry logic adapted from llm_api_integration/src/client.py
and tracking.py — copied here rather than imported, since each portfolio
project stays independently installable.

Uses google-genai (google.generativeai is deprecated as of mid-2026).
"""

import json
import logging
import os
import time

from google import genai
from google.genai import types
import mlflow

logger = logging.getLogger(__name__)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def build_model(model_name: str, temperature: float, max_output_tokens: int, json_mode: bool = False):
    config_kwargs = {"temperature": temperature, "max_output_tokens": max_output_tokens}
    if json_mode:
        config_kwargs["response_mime_type"] = "application/json"
    return model_name, types.GenerateContentConfig(**config_kwargs)


def strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text.removeprefix("json").strip()
    return text


def call_gemini(model, prompt: str, max_attempts: int = 3, backoff_seconds: int = 2):
    model_name, gen_config = model
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return client.models.generate_content(model=model_name, contents=prompt, config=gen_config)
        except Exception as e:
            last_error = e
            logger.warning(f"Gemini call failed (attempt {attempt}/{max_attempts}): {e}")
            if attempt < max_attempts:
                time.sleep(backoff_seconds * attempt)
    raise RuntimeError(f"Gemini call failed after {max_attempts} attempts") from last_error


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
    prompt = QA_PROMPT_TEMPLATE.format(n_questions=n_questions, article_text=article_text[:2000])
    response = call_gemini(model, prompt)
    raw_text = strip_markdown_fences(response.text)
    try:
        parsed = json.loads(raw_text)
        return parsed.get("qa_pairs", []), response
    except json.JSONDecodeError:
        return [], response


def compute_cost_usd(prompt_tokens: int, response_tokens: int, input_rate: float, output_rate: float) -> float:
    return (prompt_tokens / 1_000_000) * input_rate + (response_tokens / 1_000_000) * output_rate


def extract_token_counts(response) -> tuple[int, int]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return 0, 0
    return usage.prompt_token_count, usage.candidates_token_count


def init_tracking(tracking_uri: str, experiment_name: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_usage(prompt_tokens, response_tokens, model_name, input_cost_per_million=0.0, output_cost_per_million=0.0):
    cost_usd = compute_cost_usd(prompt_tokens, response_tokens, input_cost_per_million, output_cost_per_million)
    try:
        with mlflow.start_run(nested=True):
            mlflow.log_param("model", model_name)
            mlflow.log_metric("prompt_tokens", prompt_tokens)
            mlflow.log_metric("response_tokens", response_tokens)
            mlflow.log_metric("total_tokens", prompt_tokens + response_tokens)
            mlflow.log_metric("cost_usd", cost_usd)
    except Exception as e:
        logger.warning(f"MLflow logging failed, continuing without it: {e}")
    return cost_usd


def generate_qa_dataset(model, df, config: dict, sleep_seconds: float = 4.5):
    """Generate synthetic Q&A pairs for every row in df (needs id, title, article).
    Caller must run init_tracking(...) before calling this, so MLflow logging works.
    """
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
