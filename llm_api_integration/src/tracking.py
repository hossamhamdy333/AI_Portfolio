"""Logs token usage per request to MLflow.

gemini-3.1-flash-lite's free tier has no per-request dollar cost, so we log
tokens rather than dollars. The habit is what matters: in a real job you'll
need to show someone exactly how many tokens a feature is burning, and
"we log every call" is the answer they want to hear.
"""

import logging

import mlflow

logger = logging.getLogger(__name__)


def init_tracking(tracking_uri: str, experiment_name: str) -> None:
    """Point MLflow at the configured store and experiment.

    Must run once at app startup. Without it, MLflow silently defaults to
    a local ./mlruns folder and the "Default" experiment — the config
    values would exist in config.yaml but do nothing.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def compute_cost_usd(prompt_tokens: int, response_tokens: int, input_rate: float, output_rate: float) -> float:
    """Real cost math, not a hardcoded $0 — rates are per-million-tokens,
    matching how every provider (OpenAI, Anthropic, Gemini) publishes pricing.
    gemini-3.1-flash-lite tier means both rates are 0.0 by default, so this
    currently returns 0.0 — but the calculation itself is correct and ready
    for whatever model/pricing you swap in later.
    """
    return (prompt_tokens / 1_000_000) * input_rate + (response_tokens / 1_000_000) * output_rate


def log_usage(
    prompt_tokens: int,
    response_tokens: int,
    model_name: str,
    input_cost_per_million: float = 0.0,
    output_cost_per_million: float = 0.0,
) -> None:
    """Record one request's token usage and dollar cost as an MLflow run.

    Wrapped in try/except so a tracking failure (e.g. MLflow server down)
    never breaks the actual user-facing request — logging is secondary to
    serving the response.
    """
    cost_usd = compute_cost_usd(prompt_tokens, response_tokens, input_cost_per_million, output_cost_per_million)
    try:
        with mlflow.start_run():
            mlflow.log_param("model", model_name)
            mlflow.log_metric("prompt_tokens", prompt_tokens)
            mlflow.log_metric("response_tokens", response_tokens)
            mlflow.log_metric("total_tokens", prompt_tokens + response_tokens)
            mlflow.log_metric("cost_usd", cost_usd)
    except Exception as e:
        logger.warning(f"MLflow logging failed, continuing without it: {e}")


def extract_token_counts(response) -> tuple[int, int]:
    """Pull prompt/response token counts out of a Gemini response object.

    Isolated in its own function because the SDK's usage_metadata field
    location is exactly the kind of thing that changes between SDK
    versions — one place to fix it if it does.
    """
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return 0, 0
    return usage.prompt_token_count, usage.candidates_token_count
