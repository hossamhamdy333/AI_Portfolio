import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

RAW_DATA_PATH = "../data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
SCORED_DATA_PATH = "../data/scored_test.csv"
IS_FIXTURE_DATA = False

st.set_page_config(page_title="Churn Retention Dashboard", layout="wide")


@st.cache_data
def load_data():
    raw = pd.read_csv(RAW_DATA_PATH)
    raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce").fillna(0)
    scored = pd.read_csv(SCORED_DATA_PATH)
    return raw, scored


raw_df, scored_df = load_data()

st.title("Customer Churn — Retention Dashboard")
if IS_FIXTURE_DATA:
    st.warning(
        "Running on placeholder fixture data, not the real Telco dataset. "
        "Every number on this page will change once the real exports are loaded — "
        "see fixtures/scripts/ for how this was generated.",
        icon="⚠️",
    )

section = st.sidebar.radio(
    "Section",
    ["Churn Overview", "Model Calibration", "Priority Matrix", "Profit Threshold What-If"],
)

# ============================================================
# Section 1: Churn Overview
# ============================================================
if section == "Churn Overview":
    st.header("Churn Overview")

    overall_rate = (raw_df["Churn"] == "Yes").mean() * 100
    col1, col2, col3 = st.columns(3)
    col1.metric("Total customers", f"{len(raw_df):,}")
    col2.metric("Overall churn rate", f"{overall_rate:.1f}%")
    col3.metric("Avg monthly charge", f"${raw_df['MonthlyCharges'].mean():.2f}")

    st.subheader("Churn rate by contract type")
    by_contract = (
        raw_df.groupby("Contract")["Churn"]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .sort_values(ascending=False)
        .rename("Churn rate (%)")
    )
    st.bar_chart(by_contract)

    st.subheader("Churn rate by tenure band")
    tenure_bins = pd.cut(
        raw_df["tenure"], bins=[0, 12, 24, 48, 60, 72],
        labels=["0-1y", "1-2yr", "2-4yr", "4-5yr", "5-6yr"],
    )
    by_tenure = (
        raw_df.groupby(tenure_bins, observed=True)["Churn"]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .rename("Churn rate (%)")
    )
    st.bar_chart(by_tenure)

# ============================================================
# Section 2: Model Calibration
# ============================================================
elif section == "Model Calibration":
    st.header("Model Calibration")
    st.write(
        "Checks whether the model's predicted probabilities can be trusted, "
        "not just whether it ranks customers correctly. A point on the diagonal "
        "means: of the customers the model said had an X% chance of churning, "
        "about X% actually did."
    )

    from sklearn.calibration import calibration_curve

    y_true = scored_df["actual_churn"]
    y_prob = scored_df["churn_probability"]
    fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)

    calib_df = pd.DataFrame({
        "Mean predicted probability": mean_pred,
        "Fraction of actual churners": fraction_pos,
    }).set_index("Mean predicted probability")
    st.line_chart(calib_df)

    from sklearn.metrics import brier_score_loss
    brier = brier_score_loss(y_true, y_prob)
    st.metric("Brier score (lower is better)", f"{brier:.4f}")

# ============================================================
# Section 3: Priority Matrix
# ============================================================
elif section == "Priority Matrix":
    st.header("Priority Matrix")
    st.write(
        "Every test-set customer, bucketed by predicted churn risk and by "
        "monthly-charge value tier. Filter and sort to find who to call."
    )

    risk_order = ["Very Low", "Low", "Medium", "High", "Very High"]
    value_order = ["Low Value", "Mid Value", "High Value"]

    col1, col2 = st.columns(2)
    risk_filter = col1.multiselect("Risk segment", risk_order, default=risk_order)
    value_filter = col2.multiselect("Value segment", value_order, default=value_order)

    filtered = scored_df[
        scored_df["risk_segment"].isin(risk_filter)
        & scored_df["value_segment"].isin(value_filter)
    ]

    st.subheader("Customer count by segment")
    pivot = (
        filtered.pivot_table(
            index="value_segment", columns="risk_segment",
            values="customerID", aggfunc="count", observed=True,
        )
        .reindex(index=value_order, columns=risk_order)
    )
    st.dataframe(pivot.style.background_gradient(cmap="YlOrRd", axis=None))

    st.subheader(f"Customer list ({len(filtered):,} matching)")
    display_cols = [
        "customerID", "Contract", "tenure", "MonthlyCharges",
        "churn_probability", "risk_segment", "value_segment",
    ]
    st.dataframe(
        filtered[display_cols].sort_values("churn_probability", ascending=False),
        width='stretch',
    )

    st.subheader("This week's call list")
    st.write(
        "High Value customers at High or Very High risk — the group the "
        "notebook's retention strategy calls out for immediate personal outreach."
    )
    call_list = scored_df[
        (scored_df["value_segment"] == "High Value")
        & (scored_df["risk_segment"].isin(["High", "Very High"]))
    ].sort_values("churn_probability", ascending=False)
    st.dataframe(call_list[display_cols], width='stretch')

# ============================================================
# Section 4: Profit Threshold What-If
# ============================================================
else:
    st.header("Profit Threshold What-If")
    st.write(
        "The notebook picks a classification threshold by maximizing expected "
        "profit: the value of a saved customer against the cost of reaching out "
        "to one who wasn't going to churn anyway. Move the assumptions below to "
        "see how the optimal threshold shifts."
    )

    default_cltv = float(raw_df["TotalCharges"].mean())
    col1, col2, col3 = st.columns(3)
    cltv_avg = col1.slider(
        "Average customer value ($, avg TotalCharges)",
        min_value=100.0, max_value=5000.0, value=round(default_cltv, 2), step=50.0,
    )
    discount = col2.slider("Retention offer cost ($)", min_value=0, max_value=200, value=50, step=5)
    campaign_cost = col3.slider("Cost to reach a customer ($)", min_value=0, max_value=50, value=5, step=1)

    y_true = scored_df["actual_churn"].to_numpy()
    y_prob = scored_df["churn_probability"].to_numpy()

    thresholds = np.arange(0.01, 0.99, 0.01)
    profits = []
    for thresh in thresholds:
        y_pred_t = (y_prob >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_t).ravel()
        profit = (
            tp * (cltv_avg - discount - campaign_cost)
            + fp * (-discount - campaign_cost)
            + fn * (-cltv_avg)
            + tn * 0
        )
        profits.append(profit)

    profits = np.array(profits)
    optimal_idx = int(np.argmax(profits))
    optimal_threshold = thresholds[optimal_idx]
    max_profit = profits[optimal_idx]
    default_profit = profits[49]  # threshold = 0.50

    col1, col2, col3 = st.columns(3)
    col1.metric("Optimal threshold", f"{optimal_threshold:.2f}")
    col2.metric("Expected profit at optimal threshold", f"${max_profit:,.0f}")
    col3.metric("Improvement vs. 0.50 threshold", f"${max_profit - default_profit:,.0f}")

    profit_df = pd.DataFrame({"Threshold": thresholds, "Expected profit ($)": profits}).set_index("Threshold")
    st.line_chart(profit_df)

    st.caption(
        "Note: the notebook's own analysis flags that at very low thresholds "
        "the model ends up predicting churn for almost everyone, since the "
        "profit formula assumes every retention offer succeeds. Treat the "
        "optimal threshold as a starting point for discussion, not a rule to "
        "follow automatically."
    )
