"""
Streamlit dashboard for the marketing A/B test.

Conversion funnel, significance test (with a confidence-interval chart),
ROI what-if. Run with: streamlit run streamlit_app.py
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from statsmodels.stats.proportion import (
    proportions_ztest,
    proportion_confint,
    proportion_effectsize,
)

DATA_PATH = "../data/raw/marketing_AB.csv"

st.set_page_config(page_title="Marketing A/B Test", layout="wide")

# ---------- design tokens ----------
INK = "#14161F"
INK_SOFT = "#6B7280"
BORDER = "#E2E5EA"
SURFACE = "#FFFFFF"
BG = "#F5F6F8"
ACCENT = "#2F5D8A"
ACCENT_SOFT = "#EAF1F7"
GOOD = "#1F8A5F"
GOOD_SOFT = "#E7F5EE"
WARN = "#B8860B"
WARN_SOFT = "#FBF3DE"
BAD = "#B23A48"
BAD_SOFT = "#FBEAEC"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'IBM Plex Sans', sans-serif;
        color: {INK};
    }}
    .stApp {{
        background-color: {BG};
    }}
    #MainMenu, footer, header {{ visibility: hidden; }}

    .eyebrow {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: {ACCENT};
        margin-bottom: 0.3rem;
    }}
    .section-title {{
        font-size: 1.4rem;
        font-weight: 600;
        margin-top: 0;
        margin-bottom: 1.1rem;
        color: {INK};
    }}
    .hero {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 2rem 2.2rem;
        margin-bottom: 1.8rem;
    }}
    .hero-title {{
        font-size: 1.9rem;
        font-weight: 700;
        margin: 0 0 0.35rem 0;
    }}
    .hero-sub {{
        color: {INK_SOFT};
        font-size: 0.98rem;
        margin-bottom: 1.1rem;
    }}
    .hero-stat {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.1rem;
        font-weight: 600;
        color: {ACCENT};
        line-height: 1.15;
    }}
    .hero-stat-label {{
        color: {INK_SOFT};
        font-size: 0.88rem;
        margin-top: 0.2rem;
    }}
    .chip {{
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        background: {ACCENT_SOFT};
        color: {ACCENT};
        border-radius: 6px;
        padding: 0.2rem 0.55rem;
        margin-right: 0.4rem;
    }}

    .card {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
    }}
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: {SURFACE};
        border: 1px solid {BORDER} !important;
        border-radius: 12px !important;
        margin-bottom: 1.2rem;
    }}

    .verdict {{
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        font-size: 0.95rem;
        margin-top: 0.8rem;
        margin-bottom: 0.6rem;
    }}
    .verdict-good {{ background: {GOOD_SOFT}; color: {GOOD}; border: 1px solid {GOOD}22; }}
    .verdict-warn {{ background: {WARN_SOFT}; color: {WARN}; border: 1px solid {WARN}22; }}
    .verdict-bad {{ background: {BAD_SOFT}; color: {BAD}; border: 1px solid {BAD}22; }}

    div[data-testid="stMetric"] {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 0.85rem 1rem 0.7rem 1rem;
    }}
    div[data-testid="stMetricLabel"] {{
        font-size: 0.8rem;
        color: {INK_SOFT};
    }}
    div[data-testid="stMetricValue"] {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.35rem;
        color: {INK};
    }}

    .caption-note {{
        color: {INK_SOFT};
        font-size: 0.86rem;
        margin-top: 0.4rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df.drop(columns=["Unnamed: 0"], errors="ignore")


def run_ab_test(df, group_a="ad", group_b="psa", alpha=0.05):
    a = df[df["test group"] == group_a]["converted"]
    b = df[df["test group"] == group_b]["converted"]

    n_a, n_b = len(a), len(b)
    conv_a, conv_b = a.sum(), b.sum()
    rate_a, rate_b = conv_a / n_a, conv_b / n_b

    z_stat, p_value = proportions_ztest([conv_a, conv_b], [n_a, n_b], alternative="two-sided")

    ci_a = proportion_confint(conv_a, n_a, alpha=alpha, method="wilson")
    ci_b = proportion_confint(conv_b, n_b, alpha=alpha, method="wilson")

    diff = rate_a - rate_b
    se_diff = np.sqrt(rate_a * (1 - rate_a) / n_a + rate_b * (1 - rate_b) / n_b)
    diff_ci = (diff - 1.96 * se_diff, diff + 1.96 * se_diff)

    effect_size_h = proportion_effectsize(rate_a, rate_b)

    return {
        "group_a": group_a, "group_b": group_b,
        "n_a": n_a, "n_b": n_b,
        "rate_a": rate_a, "rate_b": rate_b,
        "rate_a_ci": ci_a, "rate_b_ci": ci_b,
        "diff": diff, "diff_ci": diff_ci,
        "z_stat": z_stat, "p_value": p_value,
        "significant": p_value < alpha,
        "effect_size_h": effect_size_h,
    }


def ci_chart(result):
    """Horizontal confidence-interval comparison - the actual evidence for the test result."""
    groups = [result["group_a"], result["group_b"]]
    rates = [result["rate_a"] * 100, result["rate_b"] * 100]
    los = [result["rate_a_ci"][0] * 100, result["rate_b_ci"][0] * 100]
    his = [result["rate_a_ci"][1] * 100, result["rate_b_ci"][1] * 100]
    colors = [ACCENT, INK_SOFT]

    fig = go.Figure()
    for g, r, lo, hi, c in zip(groups, rates, los, his, colors):
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[g, g],
            mode="lines",
            line=dict(color=c, width=6),
            hoverinfo="skip",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[r], y=[g],
            mode="markers",
            marker=dict(color=c, size=14, line=dict(color="white", width=2)),
            hovertemplate=f"{g}: {r:.3f}%<extra></extra>",
            showlegend=False,
        ))

    all_vals = los + his
    pad = (max(all_vals) - min(all_vals)) * 0.15
    fig.update_layout(
        height=190,
        margin=dict(l=10, r=30, t=10, b=30),
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Mono, monospace", size=12, color=INK),
        xaxis=dict(
            title="conversion rate (%)", gridcolor=BORDER, zeroline=False,
            range=[min(all_vals) - pad, max(all_vals) + pad],
        ),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    return fig


def funnel_chart(rate_by_group):
    fig = go.Figure(go.Bar(
        x=rate_by_group.index,
        y=rate_by_group["conversion_rate"] * 100,
        marker_color=[ACCENT, INK_SOFT],
        text=[f"{v:.2%}" for v in rate_by_group["conversion_rate"]],
        textposition="outside",
        hovertemplate="%{x}: %{y:.3f}%<extra></extra>",
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif", size=12, color=INK),
        yaxis=dict(title="conversion rate (%)", gridcolor=BORDER),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
        showlegend=False,
    )
    return fig


df = load_data(DATA_PATH)
result = run_ab_test(df)

abs_h = abs(result["effect_size_h"])
size_label = "small" if abs_h < 0.2 else "medium" if abs_h < 0.5 else "large"

# ---------- hero ----------
verdict_text = (
    f"Ads convert {result['diff']:.2%} points higher, "
    f"{'a statistically real gap' if result['significant'] else 'but the gap is not statistically significant'}."
)

st.markdown(
    f"""
    <div class="hero">
        <div class="eyebrow">MARKETING A/B TEST · AD vs PSA CONTROL</div>
        <div class="hero-title">Do the ads pay for themselves?</div>
        <div class="hero-sub">{verdict_text}</div>
        <div class="hero-stat">+{result['diff']:.2%} pp</div>
        <div class="hero-stat-label">conversion lift, ad vs control &nbsp;·&nbsp;
            <span class="chip">n = {result['n_a'] + result['n_b']:,}</span>
            <span class="chip">p = {result['p_value']:.2g}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- funnel ----------
st.markdown('<div class="eyebrow">OVERVIEW</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Conversion Funnel</div>', unsafe_allow_html=True)

rate_by_group = df.groupby("test group")["converted"].agg(["count", "sum", "mean"])
rate_by_group.columns = ["users", "conversions", "conversion_rate"]

with st.container(border=True):
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(
            rate_by_group.style.format({"conversion_rate": "{:.2%}"}),
            width="stretch",
        )
    with col2:
        st.plotly_chart(funnel_chart(rate_by_group), width="stretch", config={"displayModeBar": False})

# ---------- significance ----------
st.markdown('<div class="eyebrow">HYPOTHESIS TEST</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Significance</div>', unsafe_allow_html=True)

with st.container(border=True):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("p-value", f"{result['p_value']:.4g}")
    m2.metric("Difference (ad - psa)", f"{result['diff']:.3%}")
    m3.metric("95% CI on difference", f"[{result['diff_ci'][0]:.3%}, {result['diff_ci'][1]:.3%}]")
    m4.metric("Effect size (Cohen's h)", f"{result['effect_size_h']:.4f}")

    st.plotly_chart(ci_chart(result), width="stretch", config={"displayModeBar": False})

    verdict_class = "verdict-good" if result["significant"] else "verdict-warn"
    verdict_msg = (
        f"Significant at alpha=0.05 (p = {result['p_value']:.4g}). "
        f"Ad group converts {result['diff']:.3%} points higher than control."
        if result["significant"]
        else f"Not significant at alpha=0.05 (p = {result['p_value']:.4g})."
    )
    st.markdown(f'<div class="verdict {verdict_class}">{verdict_msg}</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="caption-note">Effect size (Cohen\'s h): <b>{size_label}</b>. '
        "Significant + small effect means the difference is real but may not matter practically - check the ROI panel.</div>",
        unsafe_allow_html=True,
    )

# ---------- roi ----------
st.markdown('<div class="eyebrow">ECONOMICS</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Campaign ROI (what-if)</div>', unsafe_allow_html=True)

with st.container(border=True):
    st.markdown(
        '<div class="caption-note">No cost/revenue columns in this dataset - these are adjustable assumptions.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    s1, s2 = st.columns(2)
    cost_per_impression = s1.slider(
        "Cost per ad impression ($)", min_value=0.0, max_value=0.20, value=0.02, step=0.01
    )
    revenue_per_conversion = s2.slider(
        "Revenue per conversion ($)", min_value=0.0, max_value=200.0, value=25.0, step=5.0
    )

    ad_group = df[df["test group"] == "ad"]
    psa_group = df[df["test group"] == "psa"]

    total_ad_impressions = ad_group["total ads"].sum()
    campaign_cost = total_ad_impressions * cost_per_impression

    n_ad = len(ad_group)
    actual_conversions = ad_group["converted"].sum()
    counterfactual_conversions = n_ad * psa_group["converted"].mean()

    incremental_revenue = (actual_conversions - counterfactual_conversions) * revenue_per_conversion
    roi = incremental_revenue / campaign_cost if campaign_cost > 0 else float("nan")

    r1, r2, r3 = st.columns(3)
    r1.metric("Campaign cost", f"${campaign_cost:,.2f}")
    r2.metric("Incremental revenue", f"${incremental_revenue:,.2f}")
    r3.metric("ROI", f"{roi:.2f}x")

    if roi >= 1:
        roi_class, roi_msg = "verdict-good", f"Campaign pays for itself at these assumptions ({roi:.2f}x)."
    elif roi >= 0.7:
        roi_class, roi_msg = "verdict-warn", f"Campaign is close to break-even but under 1x ({roi:.2f}x)."
    else:
        roi_class, roi_msg = "verdict-bad", f"Campaign loses money at these assumptions ({roi:.2f}x)."
    st.markdown(f'<div class="verdict {roi_class}">{roi_msg}</div>', unsafe_allow_html=True)

    if not result["significant"]:
        st.markdown(
            '<div class="caption-note">Difference above wasn\'t significant - treat this ROI as illustrative.</div>',
            unsafe_allow_html=True,
        )
