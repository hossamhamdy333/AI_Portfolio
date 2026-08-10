from pydantic import BaseModel, field_validator
from typing import Optional
import pandas as pd
import re


def arabic_ratio(text: str) -> float:
    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text))
    total_chars = max(len(text), 1)
    return arabic_chars / total_chars


class ArticleRow(BaseModel):
    id: str
    title: str
    article: str
    url: Optional[str] = None
    arabic_ratio: float

    @field_validator("article")
    @classmethod
    def article_not_empty(cls, v):
        assert len(v.strip()) > 20, "article too short to be useful"
        return v

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, v):
        assert len(v.strip()) > 0, "title is empty"
        return v

    @field_validator("arabic_ratio")
    @classmethod
    def mostly_arabic(cls, v):
        assert v >= 0.5, "article is less than 50% Arabic text"
        return v


def validate_corpus(df: pd.DataFrame):
    """Validate a raw corpus DataFrame row by row.

    Returns (clean_df, dropped_rows) where dropped_rows logs id + reason
    for every row that failed validation, instead of silently discarding it.
    """
    df = df.copy()
    if "arabic_ratio" not in df.columns:
        df["arabic_ratio"] = df["article"].apply(arabic_ratio)

    valid_rows = []
    dropped_rows = []
    cols = ["id", "title", "article", "url", "arabic_ratio"]

    for _, row in df.iterrows():
        try:
            validated = ArticleRow(**row[cols].to_dict())
            valid_rows.append(validated.model_dump())
        except Exception as e:
            dropped_rows.append({"id": row["id"], "reason": str(e)})

    return pd.DataFrame(valid_rows), dropped_rows
