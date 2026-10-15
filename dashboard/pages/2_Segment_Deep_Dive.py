"""Page 2: Segment-level SHAP Explainability & Infrastructure Breakdown."""
import sys
import os
from pathlib import Path

# Ensure workspace root and dashboard folder are on sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_DASHBOARD_DIR = Path(__file__).resolve().parent.parent
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

import streamlit as st
import pandas as pd
import plotly.express as px
import importlib
from src.models.xgboost_trainer import SeverityXGBoost
from src.explainability.shap_explainer import SeveritySHAPExplainer
from src.explainability.intervention_engine import KMDAInterventionEngine
from src.utils.feature_names import to_display_name, is_causal_actionable

try:
    from dashboard.components.disclaimer import render_disclaimer_banner, render_sidebar_disclaimer
except ImportError:
    from components.disclaimer import render_disclaimer_banner, render_sidebar_disclaimer

st.set_page_config(page_title="Segment Deep Dive | KMDA", layout="wide")

st.title("Segment Deep Dive & Explainable SHAP Analysis")
render_disclaimer_banner("shap")
render_sidebar_disclaimer()
st.markdown("Inspect local SHAP drivers behind predicted crash severity on specific Kolkata road stretches.")

model_path  = "data/processed/xgboost_severity_model.joblib"
tiers_path  = "data/processed/segment_risk_tiers.csv"
crashes_path = "data/processed/crash_level_features.parquet"

if not os.path.exists(model_path) or not os.path.exists(tiers_path) or not os.path.exists(crashes_path):
    st.error("Please run the pipeline first to train the model and generate feature datasets.")
    st.stop()

# ── Cached resource loaders (prevents multi-second reload on each widget interaction) ──
@st.cache_resource
def load_model_and_explainer():
    clf = SeverityXGBoost.load_model(model_path)
    shap_tool = SeveritySHAPExplainer(clf.model, clf.feature_names)
    engine = KMDAInterventionEngine()
    return clf, shap_tool, engine

@st.cache_data(ttl=600)
def load_data():
    return pd.read_csv(tiers_path), pd.read_parquet(crashes_path)

clf, shap_tool, engine = load_model_and_explainer()
df_segments, df_crashes = load_data()

# Sort by risk score so selectbox defaults to highest-risk segment first
df_segments_sorted = df_segments.sort_values("segment_risk_score", ascending=False)
seg_list = df_segments_sorted["segment_id"].tolist()
selected_seg_id = st.selectbox(
    "Select Segment ID to Analyze (sorted highest risk -> lowest):",
    seg_list, index=0
)

seg_data = df_segments[df_segments["segment_id"] == selected_seg_id].iloc[0]

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Corridor", seg_data.get("corridor_name", "Unknown"))
col2.metric("Priority Tier", seg_data.get("priority_tier", "N/A"))
col3.metric("Risk Score", f"{seg_data.get('segment_risk_score', 0)} / 100")
col4.metric("Total Crashes", f"{int(seg_data.get('total_crashes', 0))}")
col5.metric("Fatal Crashes", f"{int(seg_data.get('fatal_crashes', 0))}", delta_color="inverse")

st.markdown("---")

# Find crashes associated with this segment
seg_crashes = df_crashes[df_crashes["segment_id"] == selected_seg_id]

if len(seg_crashes) == 0:
    st.info(
        "No recorded crash events on this specific segment. "
        "Showing SHAP explanation from the nearest representative crash profile. "
        "This segment is **not confirmed safe** — low crash counts may reflect underreporting."
    )
    # Use most similar crash profile (match highway type and speed limit)
    highway_match = df_crashes[
        df_crashes["segment_highway_type"] == seg_data.get("highway_type", "primary")
    ]
    sample_crash = highway_match.iloc[0] if len(highway_match) > 0 else df_crashes.iloc[0]
else:
    # Use the most severe crash on this segment as the representative
    sample_crash = seg_crashes.sort_values("severity_code", ascending=False).iloc[0]

# Compute SHAP explanation
shap_drivers = shap_tool.explain_single_instance(sample_crash)
df_shap_all = pd.DataFrame(shap_drivers)

# Causal vs Correlation Filter Toggle
focus_causal = st.checkbox(
    "Focus on Actionable Engineering & Physical Drivers (Exclude Non-Causal Confounders)",
    value=True,
    help="Hides non-causal observational proxies (e.g. ambient temperature, trigonometric hour_cos/hour_sin, day of week) that correlate statistically but are not civil causes of crash severity."
)

if focus_causal:
    df_shap = df_shap_all[df_shap_all["feature"].apply(is_causal_actionable)].head(8).copy()
else:
    df_shap = df_shap_all.head(10).copy()

df_shap["display_feature"] = df_shap["feature"].map(to_display_name).fillna(df_shap["feature"])

c_left, c_right = st.columns([1.2, 1.0])

with c_left:
    title_suffix = "(Actionable Engineering & Hazard Levers)" if focus_causal else "(All Observational Features)"
    st.subheader(f"Local SHAP Drivers {title_suffix}")
    fig = px.bar(
        df_shap,
        x="shap_value",
        y="display_feature",
        orientation="h",
        color="impact",
        color_discrete_map={"Increases Risk": "#EF4444", "Decreases Risk": "#10B981"},
        labels={"shap_value": "SHAP Contribution to Fatal Crash Risk", "display_feature": "Actionable Risk Factor"},
        title=f"Feature Contributions to Fatal Risk {title_suffix}"
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Positive bars (red) increase the predicted probability of a **Fatal** outcome. "
        "Negative bars (green) reduce it. Measured in log-odds units via TreeSHAP."
    )

with c_right:
    st.subheader("Prescribed KMDA Engineering Countermeasures")
    recommended = engine.recommend_for_segment(seg_data.to_dict(), shap_drivers)
    if not recommended:
        st.success("No immediate structural civil interventions triggered for this segment.")
    for rec in recommended:
        with st.container():
            st.markdown(f"**Action:** `{rec['recommended_action']}`")
            st.caption(f"Authority: {rec['authority']} | Priority: {rec['priority']} | Cost: {rec['estimated_cost_tier']}")
            if "cmf_effect" in rec:
                st.caption(f"Causal Impact: *{rec['cmf_effect']}*")
            st.markdown("---")

# Crash breakdown for this segment
if len(seg_crashes) > 0:
    st.subheader(f"Crash Records on Segment ({len(seg_crashes)} total)")
    col_renames = {
        "timestamp": "Timestamp",
        "severity_label": "Severity",
        "primary_vehicle": "Primary Vehicle",
        "feat_is_night": "Nighttime",
        "feat_is_monsoon": "Monsoon",
        "feat_precipitation": "Precipitation (mm)"
    }
    cols_to_show = [c for c in ["timestamp", "severity_label", "primary_vehicle", "feat_is_night", "feat_is_monsoon", "feat_precipitation"] if c in seg_crashes.columns]
    st.dataframe(
        seg_crashes[cols_to_show].rename(columns=col_renames).head(20),
        use_container_width=True
    )

# City-wide Global SHAP Feature Importance
global_shap_path = "data/processed/shap_global_feature_importance.csv"
if os.path.exists(global_shap_path):
    st.markdown("---")
    st.subheader("City-Wide Risk Importance: Actionable Drivers vs. Observational Confounders")
    st.info(
        "**Causality vs. Observational Correlation in Transportation Engineering:**\n\n"
        "Machine learning models often highlight cyclical proxies like ambient temperature (°C) or trigonometric "
        "hour encodings (`hour_cos`, `hour_sin`) because they correlate statistically with unobserved conditions "
        "(e.g., late-night speeding, driver fatigue, reduced enforcement hours). "
        "However, **correlation is not causation**: 28°C vs 29°C ambient temperature does not physically cause crash fatalities. "
        "Civil engineers and police authorities can only intervene on **actionable infrastructure**—operating speeds, "
        "physical median barriers, illumination, crosswalks, and roadway friction."
    )

    df_global_all = pd.read_csv(global_shap_path)
    if focus_causal:
        df_global = df_global_all[df_global_all["feature"].apply(is_causal_actionable)].head(10).copy()
        chart_title = "Top Actionable Civil & Environmental Risk Elevators (Fatal Crashes)"
    else:
        df_global = df_global_all.head(12).copy()
        chart_title = "Top 12 All Observational Model Features (Fatal Crashes)"

    df_global["display_feature"] = df_global["feature"].map(to_display_name).fillna(df_global["feature"])
    fig_global = px.bar(
        df_global,
        x="fatal_shap_importance",
        y="display_feature",
        orientation="h",
        labels={"fatal_shap_importance": "Mean |SHAP Value| (Class: Fatal)", "display_feature": "Engineering Driver"},
        title=chart_title,
        color="fatal_shap_importance",
        color_continuous_scale="Reds"
    )
    fig_global.update_layout(yaxis=dict(autorange="reversed"), height=380, showlegend=False)
    st.plotly_chart(fig_global, use_container_width=True)

