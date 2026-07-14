
import pytest
import yaml
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import clean_text, clean_dataframe, load_config
from src.evaluate import compute_metrics
import numpy as np


#data_utils tests
def test_clean_text_lowercase():
    assert clean_text("HELLO WORLD") == "hello world"

def test_clean_text_strips_whitespace():
    assert clean_text("  hello  ") == "hello"

def test_clean_text_normalizes_spaces():
    assert clean_text("hello   world") == "hello world"

def test_clean_dataframe_drops_duplicates():
    df = pd.DataFrame({
        "text"      : ["good film", "good film", "bad film"],
        "label"     : [3, 3, 1],
        "label_name": ["positive", "positive", "negative"]
    })
    cleaned = clean_dataframe(df, min_length=1)
    assert len(cleaned) == 2

def test_clean_dataframe_drops_short():
    df = pd.DataFrame({
        "text"      : ["ok", "great film", "bad"],
        "label"     : [2, 3, 1],
        "label_name": ["neutral", "positive", "negative"]
    })
    cleaned = clean_dataframe(df, min_length=2)
    assert len(cleaned) == 1


#evaluate tests
def test_compute_metrics_perfect():
    y_true = np.array([0, 1, 2, 3, 4])
    y_pred = np.array([0, 1, 2, 3, 4])
    y_prob = np.eye(5)
    metrics = compute_metrics(y_true, y_pred, y_prob,
                              ["vn", "n", "neu", "p", "vp"])
    assert metrics["f1"] == 1.0

def test_compute_metrics_keys():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 1])
    y_prob = np.array([[0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.8,0.1]])
    metrics = compute_metrics(y_true, y_pred, y_prob,
                              ["vn", "n", "neu"])
    assert "f1" in metrics
    assert "f1_macro" in metrics
    assert "auc_roc" in metrics
