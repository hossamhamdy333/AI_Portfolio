"""
data_utils.py
─────────────
Single source of truth for all data operations in Project 01.
Every notebook imports from here — no inline data logic in notebooks.

Responsibilities:
  - Download raw datasets
  - Validate schema with Pydantic
  - Clean and normalise text
  - Stratified train/val/test split
  - Save to Parquet (processed/)
  - Generate ydata-profiling HTML report
"""

from __future__ import annotations

import re
import zipfile
import urllib.request
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml
from pydantic import BaseModel, field_validator, ValidationError
from sklearn.model_selection import train_test_split
from rich.console import Console
from rich.progress import track

console = Console()


# ── Pydantic Schemas ────────────────────────────────────────────────────────

class SMSRecord(BaseModel):
    """Schema for a single SMS Spam Collection record."""
    label: Literal["ham", "spam"]
    text: str

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        v = v.strip()
        if len(v) == 0:
            raise ValueError("text field must not be empty")
        return v


class SSTRecord(BaseModel):
    """Schema for a single SST-2 record."""
    label: int   # 0 = negative, 1 = positive
    text: str

    @field_validator("label")
    @classmethod
    def label_is_binary(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError(f"label must be 0 or 1, got {v}")
        return v

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        v = v.strip()
        if len(v) == 0:
            raise ValueError("text field must not be empty")
        return v


# ── Config Loader ───────────────────────────────────────────────────────────

def load_config(config_path: str | Path) -> dict:
    """Load and return the YAML config as a plain dict."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Downloaders ─────────────────────────────────────────────────────────────

def download_sms_spam(cfg: dict) -> pd.DataFrame:
    """
    Download the SMS Spam Collection dataset.
    Returns raw DataFrame with columns: [label, text]
    """
    raw_dir = Path(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path  = raw_dir / "smsspamcollection.zip"
    data_path = raw_dir / cfg["data"]["sms_spam"]["filename"]

    if not data_path.exists():
        console.print("[bold blue]Downloading SMS Spam Collection...[/]")
        urllib.request.urlretrieve(cfg["data"]["sms_spam"]["url"], zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(raw_dir)
        zip_path.unlink()
        console.print(f"[green]✓[/] Saved to {data_path}")
    else:
        console.print(f"[yellow]↩[/] Found cached {data_path}, skipping download")

    df = pd.read_csv(
        data_path,
        sep="\t",
        header=None,
        names=["label", "text"],
        encoding="latin-1",
    )
    return df


def download_sst2(cfg: dict) -> pd.DataFrame:
    """
    Download SST-2 from HuggingFace datasets.
    Returns raw DataFrame with columns: [label, text]
    """
    from datasets import load_dataset
    console.print("[bold blue]Loading SST-2 from HuggingFace...[/]")
    dataset = load_dataset(cfg["data"]["sst2"]["hf_name"])
    df = dataset["train"].to_pandas()[[
        cfg["data"]["sst2"]["text_col"],
        cfg["data"]["sst2"]["label_col"],
    ]].rename(columns={
        cfg["data"]["sst2"]["text_col"]:  "text",
        cfg["data"]["sst2"]["label_col"]: "label",
    })
    console.print(f"[green]✓[/] SST-2 loaded: {len(df):,} rows")
    return df


# ── Validation ──────────────────────────────────────────────────────────────

def validate_sms_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate every row against SMSRecord schema.
    Drops invalid rows and logs counts.
    """
    valid_rows, errors = [], []
    for idx, row in track(df.iterrows(), description="Validating SMS...", total=len(df)):
        try:
            r = SMSRecord(label=row["label"], text=row["text"])
            valid_rows.append({"label": r.label, "text": r.text})
        except ValidationError as e:
            errors.append({"index": idx, "error": str(e)})

    if errors:
        console.print(f"[yellow]⚠[/]  Dropped {len(errors)} invalid rows")
    console.print(f"[green]✓[/] {len(valid_rows):,} valid SMS records")
    return pd.DataFrame(valid_rows)


def validate_sst_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate every row against SSTRecord schema."""
    valid_rows, errors = [], []
    for idx, row in track(df.iterrows(), description="Validating SST-2...", total=len(df)):
        try:
            r = SSTRecord(label=int(row["label"]), text=row["text"])
            valid_rows.append({"label": r.label, "text": r.text})
        except ValidationError as e:
            errors.append({"index": idx, "error": str(e)})

    if errors:
        console.print(f"[yellow]⚠[/]  Dropped {len(errors)} invalid SST-2 rows")
    console.print(f"[green]✓[/] {len(valid_rows):,} valid SST-2 records")
    return pd.DataFrame(valid_rows)


# ── Text Cleaning ───────────────────────────────────────────────────────────

def clean_text(text: str, cfg: dict) -> str:
    """
    Apply cleaning steps defined in config.yaml → preprocessing.
    Deterministic: same input always gives same output.
    """
    pp = cfg["preprocessing"]
    if pp.get("lowercase", True):
        text = text.lower()
    if pp.get("remove_urls", True):
        text = re.sub(r"http\S+|www\S+", "", text)
    if pp.get("remove_punctuation", False):
        text = re.sub(r"[^\w\s]", "", text)
    if pp.get("strip_whitespace", True):
        text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_dataframe(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply clean_text to every row. Returns new DataFrame."""
    df = df.copy()
    df["text"] = df["text"].apply(lambda t: clean_text(t, cfg))
    before = len(df)
    df = df[df["text"].str.len() >= cfg["preprocessing"]["min_token_length"]]
    dropped = before - len(df)
    if dropped:
        console.print(f"[yellow]⚠[/]  Dropped {dropped} empty rows after cleaning")
    return df.reset_index(drop=True)


# ── Train / Val / Test Split ────────────────────────────────────────────────

def split_dataframe(
    df: pd.DataFrame, cfg: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train / val / test split.
    Returns (train_df, val_df, test_df).
    """
    splits = cfg["data"]["splits"]
    seed   = cfg["data"]["random_seed"]

    test_size = splits["test"]
    val_size  = splits["val"] / (splits["train"] + splits["val"])

    train_val, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label"]
    )
    train, val = train_test_split(
        train_val, test_size=val_size, random_state=seed, stratify=train_val["label"]
    )

    console.print(
        f"[green]✓[/] Split → "
        f"train: {len(train):,} | val: {len(val):,} | test: {len(test):,}"
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


# ── Parquet Persistence ─────────────────────────────────────────────────────

def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    cfg: dict,
    dataset_name: str,
) -> None:
    """Save train/val/test splits as Parquet files."""
    out_dir = Path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, df in [("train", train), ("val", val), ("test", test)]:
        path = out_dir / f"{dataset_name}_{split_name}.parquet"
        df.to_parquet(path, index=False)
        console.print(f"[green]✓[/] Saved {path}  ({len(df):,} rows)")


def load_split(cfg: dict, dataset_name: str, split: str) -> pd.DataFrame:
    """
    Load a single Parquet split from processed/.
    Usage: df = load_split(cfg, "sms_spam", "train")
    """
    path = Path(cfg["paths"]["processed_dir"]) / f"{dataset_name}_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed split not found: {path}\n"
            "Run notebook 01 or 'make data' first."
        )
    return pd.read_parquet(path)


# ── ydata-profiling ─────────────────────────────────────────────────────────

def run_profiling(
    cfg: dict | str | Path,
    df: pd.DataFrame | None = None,
    dataset_name: str = "sms_spam",
) -> str:
    """
    Generate a ydata-profiling HTML report.
    If df is None, loads the processed training split.
    Returns path to the generated HTML file.
    """
    if isinstance(cfg, (str, Path)):
        cfg = load_config(cfg)

    from ydata_profiling import ProfileReport

    profiles_dir = Path(cfg["paths"]["profiles_dir"])
    profiles_dir.mkdir(parents=True, exist_ok=True)

    if df is None:
        df = load_split(cfg, dataset_name, "train")

    console.print(f"[bold blue]Running ydata-profiling on {len(df):,} rows...[/]")
    profile = ProfileReport(
        df,
        title=f"P01 — {dataset_name} Training Split",
        explorative=True,
        minimal=False,
    )
    output_path = profiles_dir / f"{dataset_name}_profile.html"
    profile.to_file(output_path)
    console.print(f"[green]✓[/] Profile report → {output_path}")
    return str(output_path)


# ── Full Pipeline Entrypoint ────────────────────────────────────────────────

def download_and_validate(config_path: str | Path) -> None:
    """
    Full data pipeline: download → validate → clean → split → save parquet.
    Called by `make data`.
    """
    cfg = load_config(config_path)

    console.rule("[bold]SMS Spam Collection[/]")
    sms_raw = download_sms_spam(cfg)
    sms_val = validate_sms_dataframe(sms_raw)
    sms_cln = clean_dataframe(sms_val, cfg)
    sms_tr, sms_vl, sms_te = split_dataframe(sms_cln, cfg)
    save_splits(sms_tr, sms_vl, sms_te, cfg, "sms_spam")

    console.rule("[bold]SST-2[/]")
    sst_raw = download_sst2(cfg)
    sst_val = validate_sst_dataframe(sst_raw)
    sst_cln = clean_dataframe(sst_val, cfg)
    sst_tr, sst_vl, sst_te = split_dataframe(sst_cln, cfg)
    save_splits(sst_tr, sst_vl, sst_te, cfg, "sst2")

    console.rule("[bold green]✓ Data pipeline complete[/]")
