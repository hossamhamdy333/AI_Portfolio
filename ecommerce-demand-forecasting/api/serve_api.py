"""
FastAPI Serving Layer — Demand Forecast API

Serves the trained model behind a REST API. Feature logic is imported from
src/features.py — the same module the training notebook uses — so the API
can never silently drift from what the model was actually trained on.

Run: uvicorn api.serve_api:app --reload
Docs: http://127.0.0.1:8000/docs

pip install fastapi uvicorn
"""

from datetime import date, timedelta
from typing import Optional
import sys
import os

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from features import (
    add_calendar_features, add_lag_features, add_rolling_features,
    add_days_since_last_sale, add_price_ffill, FEATURE_COLUMNS,
)

app = FastAPI(
    title="Demand Forecast API",
    description="Serves next-N-day unit demand forecasts and stockout risk "
                 "flags per SKU, trained on the Online Retail II dataset.",
    version="1.0.0",
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "demand_model.joblib")
HISTORY_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "demand_features.csv")
SKU_LOOKUP_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "sku_lookup.csv")

model = None
history_df = None
sku_lookup_df = None


@app.on_event("startup")
def load_artifacts():
    global model, history_df, sku_lookup_df
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        raise RuntimeError(
            f"{MODEL_PATH} not found. Run notebooks/03_forecasting_models.ipynb "
            f"first — it saves the fitted model there."
        )
    history_df = pd.read_csv(HISTORY_PATH, parse_dates=["sale_date"])
    history_df = history_df.sort_values(["stock_code", "sale_date"])

    try:
        sku_lookup_df = pd.read_csv(SKU_LOOKUP_PATH).set_index("stock_code")
    except FileNotFoundError:
        raise RuntimeError(
            f"{SKU_LOOKUP_PATH} not found. Run notebooks/03_forecasting_models.ipynb "
            f"first — it saves the train-fit SKU/category lookup values there. "
            f"The API needs these exact values; recomputing them at request "
            f"time would silently disagree with what the model was trained on."
        )


class ForecastPoint(BaseModel):
    date: date
    forecast_units: float = Field(..., ge=0)


class ForecastResponse(BaseModel):
    stock_code: str
    description: Optional[str]
    forecast: list[ForecastPoint]
    stockout_risk: bool
    lead_time_days: int


class ErrorResponse(BaseModel):
    detail: str


def build_features_for_sku(sku_hist: pd.DataFrame, stock_code: str) -> pd.DataFrame:
    """
    Runs the SAME feature-building functions the training notebook uses
    (src/features.py), on this one SKU's history. This is the fix for the
    old design, which reimplemented feature logic inline here — any change
    to how a feature is computed had to be made in two places and could
    silently drift apart. Now there is exactly one implementation.
    """
    if sku_hist.empty:
        raise HTTPException(status_code=404, detail=f"No history found for stock_code '{stock_code}'.")
    if len(sku_hist) < 28:
        raise HTTPException(
            status_code=422,
            detail=f"Insufficient history for '{stock_code}' "
                   f"({len(sku_hist)} days available, 28 required for rolling features).",
        )

    df = sku_hist.copy()
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_days_since_last_sale(df)
    df = add_price_ffill(df)

    # SKU/category static features are looked up from sku_lookup.csv — the
    # exact train-fit values the model was trained on — never recomputed at
    # request time, which would silently disagree with training.
    if stock_code in sku_lookup_df.index:
        lookup_row = sku_lookup_df.loc[stock_code]
        df["sku_avg_demand_train"] = lookup_row["sku_avg_demand_train"]
        df["sku_zero_rate_train"] = lookup_row["sku_zero_rate_train"]
        df["category_freq_enc"] = lookup_row["category_freq_enc"]
    else:
        # A SKU with no history in the training set (genuinely new product).
        # Fall back to the global training averages rather than a bare 0,
        # which would look like "never sells" instead of "unknown."
        df["sku_avg_demand_train"] = sku_lookup_df["sku_avg_demand_train"].mean()
        df["sku_zero_rate_train"] = sku_lookup_df["sku_zero_rate_train"].mean()
        df["category_freq_enc"] = sku_lookup_df["category_freq_enc"].mean()

    last_row = df.iloc[[-1]][FEATURE_COLUMNS]
    return last_row


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get(
    "/forecast/{stock_code}",
    response_model=ForecastResponse,
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
def forecast(stock_code: str, horizon_days: int = 7, lead_time_days: int = 7):
    """
    Returns a day-by-day demand forecast for `stock_code` over the next
    `horizon_days`, plus a stockout-risk flag over `lead_time_days`.

    Multi-day forecasting is recursive: each subsequent day's lag/rolling
    features are rebuilt using the PREVIOUS day's forecast as if it were an
    actual. Error compounds with horizon length — a real limitation, stated
    here rather than hidden.
    """
    stock_code = stock_code.upper().strip()
    as_of = pd.Timestamp(date.today())

    working_history = history_df[history_df["stock_code"] == stock_code].copy()
    if working_history.empty:
        raise HTTPException(status_code=404, detail=f"Unknown stock_code '{stock_code}'.")

    sku_description = (
        working_history["description"].dropna().iloc[-1]
        if "description" in working_history.columns and working_history["description"].notna().any()
        else None
    )

    forecasts = []
    cursor_date = as_of
    for _ in range(horizon_days):
        feats = build_features_for_sku(working_history, stock_code)
        pred = float(np.clip(model.predict(feats)[0], 0, None))
        forecasts.append(ForecastPoint(date=cursor_date.date(), forecast_units=round(pred, 2)))

        new_row = {c: np.nan for c in working_history.columns}
        new_row.update({
            "stock_code": stock_code,
            "sale_date": cursor_date,
            "units_sold": pred,
        })
        working_history = pd.concat([working_history, pd.DataFrame([new_row])], ignore_index=True)
        cursor_date += timedelta(days=1)

    lead_time_demand = sum(f.forecast_units for f in forecasts[:lead_time_days])
    stock_proxy = working_history["units_sold"].tail(28).mean() * lead_time_days * 1.2
    at_risk = lead_time_demand > stock_proxy

    return ForecastResponse(
        stock_code=stock_code,
        description=sku_description,
        forecast=forecasts,
        stockout_risk=bool(at_risk),
        lead_time_days=lead_time_days,
    )
