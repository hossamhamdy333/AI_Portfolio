"""Load and sample the synthetic Q&A eval set.

Both implementations in this repo score themselves against the same
question set (built once, in impl_vanilla/notebooks/02_synthetic_qa.ipynb)
so their numbers are comparable in COMPARISON.md. This is the one place
that loading/sampling logic lives -- previously impl_vanilla sampled with
`qa_df.sample(n=30, ...)` inline in two different notebooks while
impl_langchain sampled with a different n through this module, so the two
implementations were silently scoring themselves on different amounts of
data. Same function, same n, same seed now -- see configs/config.yaml's
eval.n_questions.
"""

import pandas as pd


def load_eval_set(path: str) -> pd.DataFrame:
    """path: parquet file with columns question, answer, article_id, article_title."""
    return pd.read_parquet(path)


def sample_eval_set(df: pd.DataFrame, n: int, random_seed: int) -> pd.DataFrame:
    """Sample n rows deterministically. n >= len(df) just returns df as-is
    (shuffled), so callers can pass a large n to mean "use everything"
    without a separate code path.
    """
    if n >= len(df):
        return df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    return df.sample(n=n, random_state=random_seed).reset_index(drop=True)
