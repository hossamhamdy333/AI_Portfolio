"""Tests for shared/eval_set.py.

test_sample_eval_set_same_seed_gives_same_rows is the regression test for
the original bug: impl_vanilla and impl_langchain used to sample the eval
set through two different code paths with two different n values, so their
reported metrics weren't actually comparable. This test pins down the
guarantee that matters -- same df, same n, same seed always returns the
same rows -- so a future change can't silently reintroduce that.
"""

import pandas as pd

from eval_set import sample_eval_set


def _make_df(n=20):
    return pd.DataFrame({"question": [f"q{i}" for i in range(n)], "article_id": [f"a{i}" for i in range(n)]})


def test_sample_eval_set_returns_requested_n():
    df = _make_df(20)
    sample = sample_eval_set(df, n=5, random_seed=42)
    assert len(sample) == 5


def test_sample_eval_set_n_larger_than_df_returns_all_rows():
    df = _make_df(10)
    sample = sample_eval_set(df, n=100, random_seed=42)
    assert len(sample) == 10


def test_sample_eval_set_same_seed_gives_same_rows():
    df = _make_df(20)
    sample1 = sample_eval_set(df, n=5, random_seed=42)
    sample2 = sample_eval_set(df, n=5, random_seed=42)
    assert list(sample1["question"]) == list(sample2["question"])


def test_sample_eval_set_different_seed_can_give_different_rows():
    df = _make_df(20)
    sample1 = sample_eval_set(df, n=5, random_seed=1)
    sample2 = sample_eval_set(df, n=5, random_seed=2)
    assert list(sample1["question"]) != list(sample2["question"])
