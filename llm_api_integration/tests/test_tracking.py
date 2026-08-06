"""Tests for compute_cost_usd — confirms the per-million-token math is
correct, independent of whatever rate the current model happens to have."""

from src.tracking import compute_cost_usd


def test_zero_rate_gives_zero_cost():
    assert compute_cost_usd(1000, 500, input_rate=0.0, output_rate=0.0) == 0.0


def test_nonzero_rate_computes_correctly():
    # 1,000,000 prompt tokens at $1/million = $1.00
    # 500,000 response tokens at $2/million = $1.00
    cost = compute_cost_usd(1_000_000, 500_000, input_rate=1.0, output_rate=2.0)
    assert cost == 2.0


def test_init_tracking_applies_config_values():
    """Regression test for the bug where tracking_uri/experiment_name in
    config.yaml were defined but never actually applied to MLflow."""
    from unittest.mock import patch

    from src.tracking import init_tracking

    with patch("src.tracking.mlflow") as mock_mlflow:
        init_tracking("outputs/mlruns", "llm_api_integration")
        mock_mlflow.set_tracking_uri.assert_called_once_with("outputs/mlruns")
        mock_mlflow.set_experiment.assert_called_once_with("llm_api_integration")
