"""
src/features.py

Shared feature-engineering logic for the demand forecasting project.
Used by notebooks/02_feature_engineering.ipynb to build the training set -
kept as one module rather than duplicated logic so there's a single
definition of each feature.

Design rule for every feature below: at row (stock_code, date=t), a feature
may only use information available up to and including t-1 (or t for
calendar features, which are known in advance). Anything computed with a
centered or forward window is a leakage bug.
"""

import numpy as np
import pandas as pd
import holidays


def add_calendar_features(df: pd.DataFrame, date_col: str = "sale_date") -> pd.DataFrame:
    """Calendar features are known in advance — safe to use at t, not just t-1."""
    df = df.copy()
    years = range(df[date_col].dt.year.min(), df[date_col].dt.year.max() + 1)
    uk_holidays = holidays.UnitedKingdom(years=years)

    df["day_of_week"] = df[date_col].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = df[date_col].dt.month
    df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)
    df["is_uk_holiday"] = df[date_col].dt.date.isin(uk_holidays).astype(int)
    df["days_to_christmas"] = df[date_col].apply(
        lambda d: abs((pd.Timestamp(year=d.year, month=12, day=25) - d).days)
    ).clip(upper=60)
    return df


def add_lag_features(df: pd.DataFrame, target_col: str = "units_sold",
                      group_col: str = "stock_code", lags=(1, 7, 14, 28)) -> pd.DataFrame:
    """Direct lookback values — the actual demand history a forecaster has access to."""
    df = df.copy()
    grp = df.groupby(group_col)[target_col]
    for lag in lags:
        df[f"lag_{lag}d"] = grp.shift(lag)
    df["same_weekday_last_week"] = df[f"lag_7d"]  # explicit alias, used as the baseline
    return df


def add_rolling_features(df: pd.DataFrame, target_col: str = "units_sold",
                          group_col: str = "stock_code", windows=(7, 14, 28)) -> pd.DataFrame:
    """
    shift(1) FIRST, then roll, so the window at row t only ever sees t-1 and
    earlier. This is the leakage fix most tutorials skip.

    The grouped rolling window is built once per window size and mean/std
    are both read off it, instead of re-grouping and re-rolling separately
    for each statistic (the original version did the rolling computation
    three times per window — once discarded, once for mean, once for std).
    Verified to produce identical output to the original implementation.
    """
    df = df.copy()
    shifted = df.groupby(group_col)[target_col].shift(1)
    for window in windows:
        roll = shifted.groupby(df[group_col]).rolling(window, min_periods=max(3, window // 2))
        df[f"rolling_{window}d_mean"] = roll.mean().reset_index(level=0, drop=True)
        df[f"rolling_{window}d_std"] = roll.std().reset_index(level=0, drop=True)
    df["momentum_7_vs_28"] = df["rolling_7d_mean"] - df["rolling_28d_mean"]
    return df


def add_price_ffill(df: pd.DataFrame, price_col: str = "avg_unit_price",
                     group_col: str = "stock_code") -> pd.DataFrame:
    df = df.copy()
    df[price_col] = (
        df.groupby(group_col)[price_col]
        .shift(1)
        .groupby(df[group_col])
        .ffill()
    )
    return df


def add_days_since_last_sale(df: pd.DataFrame, target_col: str = "units_sold",
                              group_col: str = "stock_code") -> pd.DataFrame:
    """
    A direct intermittency signal, causal by construction (only counts
    backward from t-1). Often the single most informative feature for
    sparse demand: "sold yesterday" and "hasn't sold in 40 days" look very
    different even when their rolling averages happen to match.

    Implemented without groupby().apply() — that approach behaves
    inconsistently when the input has only one group (e.g. building
    features for a single SKU at API-request time vs. the full historical
    dataset in the training notebook), a real pandas quirk that broke this
    exact function when it was written with apply().
    """
    df = df.copy()
    had_sale = df.groupby(group_col)[target_col].shift(1).fillna(0) > 0
    streak_id = had_sale.groupby(df[group_col]).cumsum()
    df["days_since_last_sale"] = had_sale.groupby([df[group_col], streak_id]).cumcount()
    return df


def add_sku_static_features(df: pd.DataFrame, train_mask: pd.Series,
                             target_col: str = "units_sold",
                             group_col: str = "stock_code") -> pd.DataFrame:
    """
    SKU-level popularity and zero-rate, fit on TRAIN rows only (train_mask)
    then applied to the whole dataframe — same train-only discipline as
    category encoding, to avoid leaking test-period SKU behavior backward.
    """
    df = df.copy()
    train = df[train_mask]
    sku_stats = train.groupby(group_col)[target_col].agg(
        sku_avg_demand_train="mean",
        sku_zero_rate_train=lambda s: (s == 0).mean(),
    )
    global_avg = train[target_col].mean()
    global_zero_rate = (train[target_col] == 0).mean()

    df = df.merge(sku_stats, on=group_col, how="left")
    df["sku_avg_demand_train"] = df["sku_avg_demand_train"].fillna(global_avg)
    df["sku_zero_rate_train"] = df["sku_zero_rate_train"].fillna(global_zero_rate)
    return df


def add_category_encoding(df: pd.DataFrame, train_mask: pd.Series,
                           cat_col: str = "category_code") -> pd.DataFrame:
    """Frequency-encode category, fit on TRAIN only, applied to the whole frame."""
    df = df.copy()
    freq = df.loc[train_mask, cat_col].value_counts(normalize=True)
    df["category_freq_enc"] = df[cat_col].map(freq).fillna(0)
    return df


def add_market_trend_features(df: pd.DataFrame, date_col: str = "sale_date",
                               target_col: str = "units_sold",
                               group_col: str = "category_code") -> pd.DataFrame:
    """
    Per-SKU rolling means can't see a demand ramp until it's already inside
    their own 7/28-day window. This adds a CATEGORY-level trend, built at
    (category, date) then shifted by 1 day before rolling -- same causal
    discipline as add_rolling_features -- so every SKU in a category gets
    a faster-moving "is this category ramping up right now" signal than
    its own thin per-SKU history can provide alone.
    """
    df = df.copy()
    daily_cat = (
        df.groupby([group_col, date_col])[target_col]
        .sum()
        .reset_index()
        .sort_values([group_col, date_col])
    )
    shifted = daily_cat.groupby(group_col)[target_col].shift(1)
    roll7 = shifted.groupby(daily_cat[group_col]).rolling(7, min_periods=3).mean() \
        .reset_index(level=0, drop=True)
    roll28 = shifted.groupby(daily_cat[group_col]).rolling(28, min_periods=7).mean() \
        .reset_index(level=0, drop=True)
    daily_cat["category_trend_7d"] = roll7
    daily_cat["category_trend_28d"] = roll28
    daily_cat["category_trend_ratio"] = (
        daily_cat["category_trend_7d"] / daily_cat["category_trend_28d"].replace(0, np.nan)
    )
    # division-by-zero (low-volume category, 28d window genuinely at 0)
    # produces NaN here even though category_trend_28d itself is a valid,
    # non-null 0.0 -- so it survives the warm-up dropna in the notebook
    # (which only checks category_trend_28d) and trips the FEATURE_COLUMNS
    # isna() assert downstream. Fill with 1.0 ("no change") rather than
    # leaving it undefined; true warm-up rows are unaffected since those
    # already have category_trend_28d == NaN and get dropped regardless.
    daily_cat["category_trend_ratio"] = daily_cat["category_trend_ratio"].fillna(1.0)

    df = df.merge(
        daily_cat[[group_col, date_col, "category_trend_7d",
                   "category_trend_28d", "category_trend_ratio"]],
        on=[group_col, date_col], how="left",
    )
    return df


FEATURE_COLUMNS = [
    "lag_1d", "lag_7d", "lag_14d", "lag_28d",
    "rolling_7d_mean", "rolling_7d_std",
    "rolling_14d_mean", "rolling_14d_std",
    "rolling_28d_mean", "rolling_28d_std",
    "momentum_7_vs_28", "days_since_last_sale",
    "day_of_week", "is_weekend", "month", "week_of_year",
    "is_uk_holiday", "days_to_christmas",
    "avg_unit_price", "category_freq_enc",
    "sku_avg_demand_train", "sku_zero_rate_train",
    "category_trend_7d", "category_trend_28d", "category_trend_ratio",
]
