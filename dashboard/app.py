"""Main Entrypoint for the KMDA Road Safety & Hotspot Intelligence Dashboard."""
import sys
import os
import json
from pathlib import Path

# Ensure workspace root and dashboard folder are on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_DASHBOARD_DIR = Path(__file__).resolve().parent
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

try:
    from dashboard.components.disclaimer import render_disclaimer_banner, render_sidebar_disclaimer
except ImportError:
    from components.disclaimer import render_disclaimer_banner, render_sidebar_disclaimer

st.set_page_config(
    page_title="KMDA Road Safety Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Official Data Disclaimer Banner
render_disclaimer_banner("general")
render_sidebar_disclaimer()

st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 0.2rem; }
    .sub-title  { font-size: 1.05rem; color: #64748B; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">KMDA Explainable Road Accident & Hotspot Intelligence System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Spatial Risk Ranking, XGBoost Severity Analytics & SHAP-Driven Countermeasure Planning for Kolkata</div>', unsafe_allow_html=True)

tiers_path   = "data/processed/segment_risk_tiers.csv"
crashes_path = "data/processed/crash_level_features.parquet"
actions_path = "data/processed/kmda_prescribed_actions.csv"
compare_path = "data/processed/model_comparison.csv"
curves_path  = "data/processed/calibration_curves.json"

if not os.path.exists(tiers_path) or not os.path.exists(crashes_path):
    st.error("Processed pipeline data not detected. Run `python run_pipeline.py` in your terminal.")
    if st.button("Run Pipeline Now from Dashboard"):
        with st.spinner("Executing pipeline (data cleaning, spatial snapping, KDE, XGBoost, SHAP)..."):
            from run_pipeline import run
            run()
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Pipeline executed successfully!")
            st.rerun()
    st.stop()

# ── Cached Data Loaders ────────────────────────────────────────────────────
@st.cache_data(ttl=600)
def load_segments():
    return pd.read_csv(tiers_path)

@st.cache_data(ttl=600)
def load_crashes():
    return pd.read_parquet(crashes_path)

@st.cache_data(ttl=600)
def load_actions():
    return pd.read_csv(actions_path) if os.path.exists(actions_path) else pd.DataFrame()

@st.cache_data(ttl=600)
def load_comparison():
    return pd.read_csv(compare_path) if os.path.exists(compare_path) else pd.DataFrame()

@st.cache_data(ttl=600)
def load_calibration_curves():
    if os.path.exists(curves_path):
        try:
            with open(curves_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

df_segments  = load_segments()
df_crashes   = load_crashes()
df_actions   = load_actions()
df_compare   = load_comparison()
calib_data   = load_calibration_curves()

# ── Top KPI row ────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Crash Events Analyzed", f"{len(df_crashes):,}", delta="2-Year Multi-Zone Study")
with c2:
    fatal_count = (df_crashes["severity_code"] == 2).sum()
    st.metric("Fatal Crashes Identified", f"{fatal_count:,}",
              delta=f"{(fatal_count/len(df_crashes)*100):.1f}% of total", delta_color="inverse")
with c3:
    high_priority_segs = (df_segments["priority_tier"] == "High Priority").sum()
    st.metric("High-Priority Blackspot Segments", f"{high_priority_segs}",
              delta="KMDA Action Required", delta_color="inverse")
with c4:
    st.metric("Prescribed Engineering Fixes", f"{len(df_actions)}", delta="Ready for Tender")

st.markdown("---")

col_left, col_right = st.columns([1.6, 1.0])

with col_left:
    st.subheader("Crash Distribution Across Kolkata Spatial Zones")
    zone_summary = df_crashes.groupby(["spatial_zone_id", "severity_label"]).size().reset_index(name="count")
    fig_zone = px.bar(
        zone_summary, x="spatial_zone_id", y="count", color="severity_label",
        color_discrete_map={"Fatal": "#EF4444", "Grievous Injury": "#F59E0B", "Minor / Non-Injury": "#10B981"},
        barmode="stack",
        labels={"spatial_zone_id": "Planning Zone", "count": "Crash Count", "severity_label": "Severity"}
    )
    fig_zone.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=350)
    st.plotly_chart(fig_zone, use_container_width=True)

with col_right:
    st.subheader("Priority Tier Allocation")
    tier_counts = df_segments["priority_tier"].value_counts().reset_index()
    tier_counts.columns = ["Priority Tier", "Segments"]
    fig_pie = px.pie(
        tier_counts, names="Priority Tier", values="Segments",
        color="Priority Tier",
        color_discrete_map={
            "High Priority": "#DC2626",
            "Medium Priority": "#F59E0B",
            "Low Priority / Baseline": "#3B82F6",
            "Unmonitored (No Data)": "#94A3B8"
        },
        hole=0.45
    )
    fig_pie.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=350)
    st.plotly_chart(fig_pie, use_container_width=True)

# ── Production Model Validation Metrics ────────────────────────────────
if not df_compare.empty:
    st.markdown("---")
    st.subheader("Production Model Performance: Calibrated Multiclass XGBoost")
    st.caption("Evaluated with 5-Fold Spatial Grouped Cross-Validation (unseen geographic zones) and PR-AUC.")
    prod_df = df_compare[df_compare["model"].str.contains("XGBoost", case=False, na=False)].copy()
    if prod_df.empty:
        prod_df = df_compare.head(1).copy()
    display_cols = [c for c in [
        "model", "macro_f1", "fatal_recall", "quadratic_weighted_kappa",
        "pr_auc_fatal", "pr_auc_grievous", "brier_fatal"
    ] if c in prod_df.columns]
    col_rename = {
        "model": "Model Architecture",
        "macro_f1": "Macro-F1 (Spatial CV)",
        "fatal_recall": "Fatal Recall",
        "quadratic_weighted_kappa": "Quadratic Weighted Kappa",
        "pr_auc_fatal": "Fatal PR-AUC",
        "pr_auc_grievous": "Grievous PR-AUC",
        "brier_fatal": "Brier Score (Fatal)"
    }
    st.dataframe(prod_df[display_cols].rename(columns=col_rename), use_container_width=True)

# ── Calibration Curves (Reliability Diagram) ────────────────────────────────
if calib_data:
    st.markdown("---")
    st.subheader("Probability Calibration Curves (Reliability Diagram)")
    st.caption("Comparison of predicted probabilities against empirical fraction of positive crash events (5 uniform bins).")
    fig_calib = go.Figure()
    # Perfect calibration reference line
    fig_calib.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Perfect Calibration",
        line=dict(color="#94A3B8", dash="dash")
    ))
    color_map = {"fatal": "#EF4444", "grievous": "#F59E0B", "minor": "#10B981"}
    for label, coords in calib_data.items():
        if "mean_predicted" in coords and "fraction_positives" in coords:
            fig_calib.add_trace(go.Scatter(
                x=coords["mean_predicted"],
                y=coords["fraction_positives"],
                mode="lines+markers",
                name=f"{label.capitalize()} Class",
                line=dict(color=color_map.get(label, "#3B82F6"), width=2),
                marker=dict(size=7)
            ))
    fig_calib.update_layout(
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Empirical Fraction of Positives",
        height=380,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig_calib, use_container_width=True)

st.info("**Navigation:** Use the sidebar to explore the **Hotspot Map**, **Segment SHAP Deep-Dive**, **What-If Simulator**, or the **KMDA Policy Brief**.")

if st.sidebar.button("Clear Cache & Reload Data"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()
