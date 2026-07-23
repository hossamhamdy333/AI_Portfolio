"""Shared eval set loader.

Each implementation has its own synthetic Q&A pairs (different corpus for
impl_llamaindex), but all three load them through this same function so the
loading/format logic can't quietly diverge between implementations.
"""

import pandas as pd


def load_eval_set(parquet_path):
    """Load synthetic Q&A pairs.

    Expected columns: question, answer, article_id (or equivalent source id).
    Returns a pandas DataFrame.
    """
    return pd.read_parquet(parquet_path)


def sample_eval_set(qa_df, n, random_seed):
    """Draw a reproducible sample for evaluation, same seed convention as vanilla."""
    return qa_df.sample(n=n, random_state=random_seed)
