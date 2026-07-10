import os
import re
import yaml
import pandas as pd
from datasets import load_dataset
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load SST-5 from HuggingFace Hub.
    Returns train, val, test DataFrames.
    """
    raw = load_dataset("SetFit/sst5")
    train_df = raw["train"].to_pandas()
    val_df   = raw["validation"].to_pandas()
    test_df  = raw["test"].to_pandas()

    # Rename label_text to label_name for consistency
    for df in [train_df, val_df, test_df]:
        df.rename(columns={"label_text": "label_name"}, inplace=True)

    return train_df, val_df, test_df


def clean_text(text: str) -> str:
    """
    Clean a single sentence.
    Decisions based on EDA findings:
    - Strip leading/trailing whitespace
    - Normalize multiple spaces to one
    - Keep punctuation — it carries sentiment signal
    - Lowercase
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.lower()
    return text


def clean_dataframe(df: pd.DataFrame, min_length: int = 3) -> pd.DataFrame:
    """
    Clean a DataFrame:
    - Drop duplicate sentences (EDA found 10 in train)
    - Drop sentences shorter than min_length words
    - Apply text cleaning
    """
    original_size = len(df)

    # Drop duplicates
    df = df.drop_duplicates(subset="text").reset_index(drop=True)

    # Clean text
    df["text"] = df["text"].apply(clean_text)

    # Drop short sentences
    df["word_count"] = df["text"].str.split().str.len()
    df = df[df["word_count"] >= min_length].reset_index(drop=True)

    print(f"  Cleaned: {original_size} → {len(df)} rows "
          f"(dropped {original_size - len(df)})")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic features used in EDA and analysis."""
    df["word_count"] = df["text"].str.split().str.len()
    df["char_count"] = df["text"].str.len()
    return df


def load_and_clean(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: load → clean → add features.
    This is the main function called by notebooks.
    """
    print("Loading raw data...")
    train_df, val_df, test_df = load_raw_data()

    min_len = config["data"]["min_length"]

    print("Cleaning splits:")
    print("  Train:")
    train_df = clean_dataframe(train_df, min_length=min_len)
    print("  Validation:")
    val_df   = clean_dataframe(val_df,   min_length=min_len)
    print("  Test:")
    test_df  = clean_dataframe(test_df,  min_length=min_len)

    print("Adding features...")
    train_df = add_features(train_df)
    val_df   = add_features(val_df)
    test_df  = add_features(test_df)

    print(f"\nFinal sizes:")
    print(f"  Train      : {len(train_df)}")
    print(f"  Validation : {len(val_df)}")
    print(f"  Test       : {len(test_df)}")

    return train_df, val_df, test_df


def save_parquet(train_df, val_df, test_df, data_dir: str):
    """Save cleaned DataFrames to Parquet."""
    os.makedirs(data_dir, exist_ok=True)
    train_df.to_parquet(f"{data_dir}/train.parquet", index=False)
    val_df.to_parquet(f"{data_dir}/val.parquet",     index=False)
    test_df.to_parquet(f"{data_dir}/test.parquet",   index=False)
    print(f"\nSaved parquet files to {data_dir}")


def load_parquet(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load cleaned DataFrames from Parquet."""
    train_df = pd.read_parquet(f"{data_dir}/train.parquet")
    val_df   = pd.read_parquet(f"{data_dir}/val.parquet")
    test_df  = pd.read_parquet(f"{data_dir}/test.parquet")
    return train_df, val_df, test_df
