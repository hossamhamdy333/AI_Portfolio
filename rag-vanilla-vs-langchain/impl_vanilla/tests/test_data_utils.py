"""Tests for src/data_utils.py -- Pydantic validation logic."""

import pandas as pd
from src.data_utils import validate_corpus, arabic_ratio


def test_arabic_ratio_all_arabic():
    # A real Arabic sentence includes spaces between words, which aren't in
    # the Arabic unicode range, so a normal sentence lands well under 1.0 --
    # this used to assert > 0.9, which even genuinely all-Arabic prose
    # fails (01_eda.ipynb measured a corpus-wide mean of 0.81). > 0.7 is a
    # threshold an all-Arabic sentence can actually clear.
    text = "هذا نص عربي بالكامل"
    assert arabic_ratio(text) > 0.7


def test_arabic_ratio_all_english():
    text = "This is entirely English text"
    assert arabic_ratio(text) < 0.1


def test_validate_corpus_keeps_valid_rows():
    df = pd.DataFrame([
        {"id": "1", "title": "عنوان صحيح", "article": "هذا مقال عربي طويل بما فيه الكفاية ليمر التحقق", "url": "http://x.com"},
    ])
    clean_df, dropped = validate_corpus(df)
    assert len(clean_df) == 1
    assert len(dropped) == 0


def test_validate_corpus_drops_short_article():
    df = pd.DataFrame([
        {"id": "1", "title": "عنوان", "article": "قصير", "url": None},
    ])
    clean_df, dropped = validate_corpus(df)
    assert len(clean_df) == 0
    assert len(dropped) == 1
    assert "too short" in dropped[0]["reason"]


def test_validate_corpus_drops_non_arabic():
    df = pd.DataFrame([
        {"id": "1", "title": "Title", "article": "This is a fully English article with plenty of words in it", "url": None},
    ])
    clean_df, dropped = validate_corpus(df)
    assert len(clean_df) == 0
    assert len(dropped) == 1
    assert "Arabic" in dropped[0]["reason"]
