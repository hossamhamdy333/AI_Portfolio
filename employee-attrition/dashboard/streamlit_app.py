"""
Employee Attrition & Retention Risk Dashboard

Reads the outputs already produced by the notebooks — raw HR data and the
exported model risk scores — rather than retraining anything itself. Single
source of truth: if the numbers here look different from the notebooks,
that's a bug to fix, not two versions of the truth to reconcile.

Run with: streamlit run dashboard/streamlit_app.py
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------
# Page config & style
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Attrition & Retention Risk",
    page_icon="📊",
    layout="wide",
)

RISK_COLOR = "#C44E52"
SAFE_COLOR = "#4C72B0"
NEUTRAL_COLOR = "#8C8C8C"


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # project root, regardless of cwd

@st.cache_data
def load_data():
    raw = pd.read_csv(BASE_DIR / "data" / "raw" / "WA_Fn-UseC_-HR-Employee-Attrition.csv", encoding="utf-8")
    risk = pd.read_csv(BASE_DIR / "data" / "processed" / "fact_risk_scores.csv", encoding="utf-8")
    df = raw.merge(
        risk[["employee_id", "attrition_risk_score", "expected_attrition_cost", "replacement_cost"]],
        left_on="EmployeeNumber", right_on="employee_id", how="left",
    )
    return df


try:
    df = load_data()
except FileNotFoundError as e:
    st.error(
        "Missing data file. Run `01_eda.ipynb` through `05_cost_of_attrition.ipynb` "
        "first — this dashboard reads their outputs, it doesn't compute anything itself.\n\n"
        f"Detail: {e}"
    )
    st.stop()


# ---------------------------------------------------------------------
# Sidebar filters — shared across all pages
# ---------------------------------------------------------------------
st.sidebar.title("Filters")
dept_filter = st.sidebar.multiselect(
    "Department", options=sorted(df["Department"].unique()), default=None
)
role_filter = st.sidebar.multiselect(
    "Job Role", options=sorted(df["JobRole"].unique()), default=None
)
overtime_filter = st.sidebar.radio("OverTime", options=["All", "Yes", "No"], index=0)

filtered = df.copy()
if dept_filter:
    filtered = filtered[filtered["Department"].isin(dept_filter)]
if role_filter:
    filtered = filtered[filtered["JobRole"].isin(role_filter)]
if overtime_filter != "All":
    filtered = filtered[filtered["OverTime"] == overtime_filter]

if filtered.empty:
    st.warning("No employees match the current filters.")
    st.stop()

page = st.sidebar.radio(
    "Page", ["Executive Summary", "Attrition Drivers", "Tenure & Survival", "Cost & Retention ROI"]
)

st.title("Employee Attrition & Retention Risk")
st.caption(f"Showing {len(filtered):,} of {len(df):,} employees" + (" (filtered)" if len(filtered) != len(df) else ""))


# ---------------------------------------------------------------------
# Page 1 — Executive Summary
# ---------------------------------------------------------------------
if page == "Executive Summary":
    attrition_rate = (filtered["Attrition"] == "Yes").mean()
    total_cost = filtered["expected_attrition_cost"].sum()
    avg_risk = filtered["attrition_risk_score"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Headcount", f"{len(filtered):,}")
    c2.metric("Attrition Rate", f"{attrition_rate:.1%}")
    c3.metric("Expected Attrition Cost", f"${total_cost:,.0f}")
    c4.metric("Avg. Risk Score", f"{avg_risk:.1%}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        dept_attr = (
            filtered.groupby("Department")["Attrition"]
            .apply(lambda s: (s == "Yes").mean())
            .sort_values(ascending=False)
            .reset_index(name="attrition_rate")
        )
        fig = px.bar(
            dept_attr, x="Department", y="attrition_rate",
            title="Which departments lose the most people?",
            color_discrete_sequence=[RISK_COLOR],
        )
        fig.update_yaxes(tickformat=".0%", title="Attrition rate")
        st.plotly_chart(fig, width='stretch')

    with col2:
        role_attr = (
            filtered.groupby("JobRole")["Attrition"]
            .apply(lambda s: (s == "Yes").mean())
            .sort_values(ascending=False)
            .head(10)
            .reset_index(name="attrition_rate")
        )
        fig = px.bar(
            role_attr, x="attrition_rate", y="JobRole", orientation="h",
            title="Which roles leave fastest?",
            color_discrete_sequence=[RISK_COLOR],
        )
        fig.update_xaxes(tickformat=".0%", title="Attrition rate")
        fig.update_yaxes(title="", categoryorder="total ascending")
        st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------
# Page 2 — Attrition Drivers
# ---------------------------------------------------------------------
elif page == "Attrition Drivers":
    st.subheader("Does overtime compound with the highest-risk roles?")

    interaction = (
        filtered.groupby(["JobRole", "OverTime"])["Attrition"]
        .apply(lambda s: (s == "Yes").mean())
        .unstack()
        .fillna(0)
    )
    interaction = interaction.reindex(columns=["No", "Yes"], fill_value=0)

    fig = go.Figure()
    fig.add_bar(name="No overtime", x=interaction.index, y=interaction["No"], marker_color=SAFE_COLOR)
    fig.add_bar(name="Overtime", x=interaction.index, y=interaction["Yes"], marker_color=RISK_COLOR)
    fig.update_layout(barmode="group", yaxis_tickformat=".0%", yaxis_title="Attrition rate")
    st.plotly_chart(fig, width='stretch')
    st.caption(
        "Confirmed finding: overtime and job role compound. In the full dataset, "
        "Sales Representatives working overtime reach 66.7% attrition — far above either factor alone."
    )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(
            filtered, x="MonthlyIncome", y="attrition_risk_score", color="Department",
            title="Income vs. model risk score",
            labels={"attrition_risk_score": "Risk score", "MonthlyIncome": "Monthly income"},
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, width='stretch')

    with col2:
        sat_cols = ["JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction", "WorkLifeBalance"]
        rows = []
        for col in sat_cols:
            rate_by_level = filtered.groupby(col)["Attrition"].apply(lambda s: (s == "Yes").mean())
            for level, rate in rate_by_level.items():
                rows.append({"measure": col, "level": level, "attrition_rate": rate})
        sat_df = pd.DataFrame(rows)
        fig = px.line(
            sat_df, x="level", y="attrition_rate", color="measure",
            title="Attrition rate by satisfaction score (1=low, 4=high)",
            markers=True,
        )
        fig.update_yaxes(tickformat=".0%", title="Attrition rate")
        st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------
# Page 3 — Tenure & Survival
# ---------------------------------------------------------------------
elif page == "Tenure & Survival":
    st.subheader("Attrition risk by tenure")
    st.caption(
        "Approximate view (bucketed attrition rate, not a true Kaplan-Meier curve — "
        "see `04_survival_analysis.ipynb` for the actual survival model and Cox hazard ratios)."
    )

    bucketed = filtered.copy()
    bucketed["tenure_bucket"] = pd.cut(
        bucketed["YearsAtCompany"],
        bins=[-1, 1, 4, 9, 100],
        labels=["0-1 yrs", "2-4 yrs", "5-9 yrs", "10+ yrs"],
    )
    tenure_attr = (
        bucketed.groupby("tenure_bucket", observed=True)["Attrition"]
        .apply(lambda s: (s == "Yes").mean())
        .reset_index(name="attrition_rate")
    )
    fig = px.bar(
        tenure_attr, x="tenure_bucket", y="attrition_rate",
        title="Attrition rate drops sharply after year one",
        color_discrete_sequence=[RISK_COLOR],
    )
    fig.update_yaxes(tickformat=".0%", title="Attrition rate")
    st.plotly_chart(fig, width='stretch')

    col1, col2 = st.columns(2)
    leavers_tenure = filtered.loc[filtered["Attrition"] == "Yes", "YearsAtCompany"].median()
    stayers_tenure = filtered.loc[filtered["Attrition"] == "No", "YearsAtCompany"].median()
    col1.metric("Median tenure — Leavers", f"{leavers_tenure:.0f} yrs")
    col2.metric("Median tenure — Stayers", f"{stayers_tenure:.0f} yrs")

    st.divider()
    st.subheader("Cox model hazard ratios (from `04_survival_analysis.ipynb`)")
    hazard_df = pd.DataFrame({
        "Factor": ["OverTime", "JobSatisfaction", "DistanceFromHome", "MonthlyIncome"],
        "Hazard Ratio": [3.19, 0.79, 1.02, 1.00],
        "Interpretation": [
            "~3.2x faster attrition, holding other factors constant",
            "Each satisfaction point cuts risk ~21%",
            "Small effect, ~2% higher risk per mile",
            "Statistically significant, but tiny effect per dollar",
        ],
    })
    st.dataframe(hazard_df, width='stretch', hide_index=True)


# ---------------------------------------------------------------------
# Page 4 — Cost & Retention ROI
# ---------------------------------------------------------------------
elif page == "Cost & Retention ROI":
    total_cost = filtered["expected_attrition_cost"].sum()

    st.subheader("Where the cost concentrates")
    cost_by_dept = (
        filtered.groupby("Department")["expected_attrition_cost"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    fig = px.bar(
        cost_by_dept, x="Department", y="expected_attrition_cost",
        title="Expected annual attrition cost by department",
        color_discrete_sequence=[RISK_COLOR],
    )
    fig.update_yaxes(title="Expected cost ($)", tickprefix="$")
    st.plotly_chart(fig, width='stretch')

    st.divider()
    st.subheader("Retention intervention — what-if")
    st.caption(
        "Targets employees who are both high-risk (top 20% by model score) AND working "
        "overtime — the group the survival analysis flagged as highest-hazard."
    )

    col1, col2, col3 = st.columns(3)
    intervention_cost = col1.slider("Cost per employee ($)", 500, 5000, 2000, step=250)
    risk_reduction = col2.slider("Assumed risk reduction (%)", 0, 80, 30, step=5) / 100
    risk_pctile = col3.slider("Target top N% by risk", 5, 50, 20, step=5) / 100

    risk_threshold = filtered["attrition_risk_score"].quantile(1 - risk_pctile)
    target_group = filtered[
        (filtered["attrition_risk_score"] >= risk_threshold) & (filtered["OverTime"] == "Yes")
    ]

    n_targeted = len(target_group)
    total_intervention_cost = n_targeted * intervention_cost
    expected_cost_before = target_group["expected_attrition_cost"].sum()
    expected_cost_after = expected_cost_before * (1 - risk_reduction)
    expected_savings = expected_cost_before - expected_cost_after
    roi = (expected_savings - total_intervention_cost) / total_intervention_cost if total_intervention_cost > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Employees targeted", f"{n_targeted}")
    c2.metric("Intervention cost", f"${total_intervention_cost:,.0f}")
    c3.metric("Expected savings", f"${expected_savings:,.0f}")
    c4.metric("ROI", f"{roi:.0%}")

    st.caption(
        "All figures on this page depend on the assumptions above (cost per employee, "
        "assumed risk reduction) — adjust the sliders to see how sensitive the ROI is. "
        "The department cost chart is the only number here derived purely from the "
        "validated model, not an assumption."
    )
