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

try:
    from dashboard.components.ui_cards import (
        inject_custom_css, render_infographic_kpi, create_radial_gauge, render_road_profile_cards
    )
except ImportError:
    from components.ui_cards import (
        inject_custom_css, render_infographic_kpi, create_radial_gauge, render_road_profile_cards
    )

st.set_page_config(page_title="Segment Deep Dive | KMDA", layout="wide")
inject_custom_css()

st.title("Segment Deep Dive & Explainable SHAP Analysis")
render_disclaimer_banner("shap")
render_sidebar_disclaimer()
st.markdown("Inspect local TreeSHAP drivers behind predicted crash severity on specific Kolkata road stretches.")

model_path  = "data/processed/xgboost_severity_model.joblib"
tiers_path  = "data/processed/segment_risk_tiers.csv"
crashes_path = "data/processed/crash_level_features.parquet"

if not os.path.exists(model_path) or not os.path.exists(tiers_path) or not os.path.exists(crashes_path):
    st.error("Please run the pipeline first to train the model and generate feature datasets.")
    st.stop()

# ── Cached resource loaders ────────────────────────────────────────────────
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

# ── Top Infographic Header & Radial Risk Speedometer ───────────────────────
hdr_left, hdr_right = st.columns([1.5, 1.0])

with hdr_left:
    tier = str(seg_data.get("priority_tier", "Low Priority"))
    tier_badge_color = "badge-red" if "High" in tier else ("badge-amber" if "Medium" in tier else "badge-blue")
    rank_val = seg_data.get("rank", "N/A")

    st.markdown(f"""
    <div class="kmda-box" style="margin-bottom: 12px;">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
            <span class="kmda-badge {tier_badge_color}">{tier}</span>
            <span style="font-size: 0.80rem; font-weight: 700; color: var(--kmda-text-muted);">CITY RANK: #{rank_val}</span>
        </div>
        <div style="font-size: 1.55rem; font-weight: 800; font-family: 'Outfit', sans-serif; color: var(--kmda-text-primary); margin-bottom: 4px;">
            {seg_data.get("corridor_name", "Unknown Corridor")}
        </div>
        <div style="font-size: 0.85rem; color: var(--kmda-text-muted); margin-bottom: 14px;">
            <b>Segment ID:</b> <code>{selected_seg_id}</code> | <b>Zone:</b> {seg_data.get("spatial_zone_id", "N/A")} | <b>Length:</b> {int(seg_data.get("length_meters", 350))}m
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Physical cross-section cards
    render_road_profile_cards(
        speed_limit=float(seg_data.get("speed_limit", 50)),
        lanes=float(seg_data.get("lanes", 4)),
        has_median=int(seg_data.get("has_median", 0)),
        is_lit=int(seg_data.get("is_lit", 1))
    )

with hdr_right:
    risk_score = float(seg_data.get("segment_risk_score", 0.0))
    fig_gauge = create_radial_gauge(risk_score, title="Composite Risk Score (0-100)", height=225)
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Mini crash summary metrics
    tot_c = int(seg_data.get("total_crashes", 0))
    fat_c = int(seg_data.get("fatal_crashes", 0))
    ped_c = int(seg_data.get("pedestrian_crashes", 0))
    st.markdown(f"""
    <div style="display: flex; gap: 8px; justify-content: center; margin-top: -15px;">
        <span class="kmda-badge badge-slate">Total: {tot_c}</span>
        <span class="kmda-badge badge-red">Fatal: {fat_c}</span>
        <span class="kmda-badge badge-amber">VRU/Ped: {ped_c}</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="kmda-divider"></div>', unsafe_allow_html=True)

# Find crashes associated with this segment
seg_crashes = df_crashes[df_crashes["segment_id"] == selected_seg_id]

if len(seg_crashes) == 0:
    st.info(
        "No recorded crash events on this specific segment. "
        "Showing SHAP explanation from the nearest representative crash profile. "
        "This segment is **not confirmed safe** — low crash counts may reflect underreporting."
    )
    highway_match = df_crashes[
        df_crashes["segment_highway_type"] == seg_data.get("highway_type", "primary")
    ]
    sample_crash = highway_match.iloc[0] if len(highway_match) > 0 else df_crashes.iloc[0]
else:
    sample_crash = seg_crashes.sort_values("severity_code", ascending=False).iloc[0]

# Compute SHAP explanation
shap_drivers = shap_tool.explain_single_instance(sample_crash)
df_shap_all = pd.DataFrame(shap_drivers)

# Causal vs Correlation Filter Toggle
focus_causal = st.checkbox(
    "🎯 Focus on Actionable Engineering & Physical Drivers (Exclude Non-Causal Confounders)",
    value=True,
    help="Hides non-causal observational proxies (e.g. ambient temperature, trigonometric hour_cos/hour_sin, day of week) that correlate statistically but are not civil causes of crash severity."
)

if focus_causal:
    df_shap = df_shap_all[df_shap_all["feature"].apply(is_causal_actionable)].head(8).copy()
else:
    df_shap = df_shap_all.head(10).copy()

df_shap["display_feature"] = df_shap["feature"].map(to_display_name).fillna(df_shap["feature"])

c_left, c_right = st.columns([1.25, 1.0])

with c_left:
    title_suffix = "(Actionable Civil Levers)" if focus_causal else "(All Observational Features)"
    st.subheader(f"📊 Local SHAP Drivers {title_suffix}")
    fig = px.bar(
        df_shap,
        x="shap_value",
        y="display_feature",
        orientation="h",
        color="impact",
        color_discrete_map={"Increases Risk": "#EF4444", "Decreases Risk": "#10B981"},
        labels={"shap_value": "SHAP Contribution to Fatal Crash Risk (Log-Odds)", "display_feature": "Identified Risk Factor"},
        title=f"Feature Contributions to Predicted Severity"
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        height=380,
        margin=dict(l=15, r=20, t=35, b=15),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Inter, sans-serif")
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "🔴 **Red bars** push the predicted probability toward **Fatal / Grievous**. "
        "🟢 **Green bars** exert a mitigating protective effect."
    )

with c_right:
    st.subheader("🛠️ Prescribed KMDA Engineering Countermeasures")
    recommended = engine.recommend_for_segment(seg_data.to_dict(), shap_drivers)
    if not recommended:
        st.success("No immediate structural civil interventions triggered for this segment.")
    for rec in recommended:
        prio_color = "act-high" if rec['priority'] == "High" else ("act-med" if rec['priority'] == "Medium" else "act-low")
        cost_badge_color = "badge-red" if "High" in rec['estimated_cost_tier'] else ("badge-amber" if "Medium" in rec['estimated_cost_tier'] else "badge-green")

        st.markdown(f"""
        <div class="kmda-action-card {prio_color}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span class="kmda-badge badge-blue">{rec['authority']}</span>
                <span class="kmda-badge {cost_badge_color}">Cost: {rec['estimated_cost_tier']}</span>
            </div>
            <div style="font-weight: 700; font-size: 0.92rem; color: var(--kmda-text-primary); margin-bottom: 4px;">
                {rec['recommended_action']}
            </div>
            <div style="font-size: 0.78rem; color: var(--kmda-text-muted);">
                <b>Expected Safety Impact:</b> <i>{rec.get('cmf_effect', 'Reduces fatal conflict exposure')}</i>
            </div>
        </div>
        """, unsafe_allow_html=True)

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
    fig_global.update_layout(
        yaxis=dict(autorange="reversed"),
        height=380,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif")
    )
    st.plotly_chart(fig_global, use_container_width=True)

