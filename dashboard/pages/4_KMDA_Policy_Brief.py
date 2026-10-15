"""Page 4: KMDA Official Executive Policy Brief & Action Plan."""
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

try:
    from dashboard.components.disclaimer import render_disclaimer_banner, render_sidebar_disclaimer
except ImportError:
    from components.disclaimer import render_disclaimer_banner, render_sidebar_disclaimer

st.set_page_config(page_title="KMDA Policy Brief", layout="wide")

st.title("KMDA Policy Brief: Priority Road Safety Interventions")
render_disclaimer_banner("policy")
render_sidebar_disclaimer()
st.markdown("""
**To:** Directorate of Infrastructure & Urban Mobility, Kolkata Metropolitan Development Authority (KMDA)  
**Subject:** Ranked Road Blackspot Stretches & Prescribed Civil Countermeasures  
**Status:** Prototype — Synthetic Training Data
""")

blackspots_path = "data/processed/kmda_top20_blackspots.csv"
actions_path    = "data/processed/kmda_prescribed_actions.csv"
compare_path    = "data/processed/model_comparison.csv"
shap_path       = "data/processed/shap_global_feature_importance.csv"

if not os.path.exists(blackspots_path) or not os.path.exists(actions_path):
    st.error("Please run the pipeline to generate blackspot and intervention records.")
    st.stop()

@st.cache_data(ttl=600)
def load_brief_data():
    top20   = pd.read_csv(blackspots_path)
    actions = pd.read_csv(actions_path)
    compare = pd.read_csv(compare_path) if os.path.exists(compare_path) else pd.DataFrame()
    shap_df = pd.read_csv(shap_path) if os.path.exists(shap_path) else pd.DataFrame()
    return top20, actions, compare, shap_df

df_top20, df_actions, df_compare, df_shap = load_brief_data()

st.subheader("1. Executive Summary (Prototype Findings)")
st.markdown("""
Under the spatial risk analysis protocol, road segments across the Kolkata metropolitan area were evaluated using
**1D Network-Constrained Kernel Density Estimation (NetKDE)** and **Spatial Grouped Cross-Validation with XGBoost & SHAP**
with Isotonic probability calibration.

The synthetic-data prototype reveals three primary **simulated risk drivers** (to be validated with real data):
1. **Deficient Nighttime Illumination:** Unlit stretches along high-speed corridors show elevated modelled fatal odds.
2. **Speed-Profile Incongruence:** 60+ km/h segments with mixed non-motorized traffic exposure.
3. **Unprotected Junction Crossings:** Absence of grade-separated or tabletop pedestrian crosswalks.

**Production Model Architecture:**
- **Multiclass XGBoost** with 5-Fold Spatial Grouped Cross-Validation and Isotonic Probability Calibration.
""")

if not df_compare.empty:
    prod_df = df_compare[df_compare["model"].str.contains("XGBoost", case=False, na=False)].copy()
    if prod_df.empty:
        prod_df = df_compare.head(1).copy()
    display_cols = [c for c in [
        "model", "macro_f1", "fatal_recall", "quadratic_weighted_kappa",
        "pr_auc_fatal", "brier_fatal"
    ] if c in prod_df.columns]
    col_rename = {
        "model": "Model Architecture",
        "macro_f1": "Macro-F1 (Spatial CV)",
        "fatal_recall": "Fatal Recall",
        "quadratic_weighted_kappa": "Quadratic Weighted Kappa",
        "pr_auc_fatal": "Fatal PR-AUC",
        "brier_fatal": "Brier Score (Fatal)"
    }
    st.dataframe(prod_df[display_cols].rename(columns=col_rename), use_container_width=True)

from src.utils.feature_names import to_display_name

if not df_shap.empty:
    st.markdown("**Top Simulated Risk Drivers for Fatal Crashes across Kolkata:**")
    df_shap_display = df_shap.head(8).copy()
    df_shap_display["feature"] = df_shap_display["feature"].map(to_display_name).fillna(df_shap_display["feature"])
    st.dataframe(
        df_shap_display.rename(columns={
            "feature": "Risk Factor",
            "fatal_shap_importance": "Mean |SHAP| Severity Weight"
        }),
        use_container_width=True
    )

st.subheader("2. Top-20 Priority Blackspot Stretches")
display_seg_cols = [c for c in [
    "rank", "segment_id", "corridor_name", "spatial_zone_id",
    "segment_risk_score", "crash_rate_per_km", "total_crashes", "fatal_crashes",
    "grievous_crashes", "speed_limit", "lanes"
] if c in df_top20.columns]
st.dataframe(df_top20[display_seg_cols], use_container_width=True)

st.subheader("3. Prescribed Municipal Engineering Actions")
st.dataframe(
    df_actions[[
        "corridor_name", "segment_id", "priority_tier",
        "intervention_id", "authority", "action", "estimated_cost_tier"
    ]],
    use_container_width=True
)

# Download buttons
c1, c2 = st.columns(2)
with c1:
    csv_top20 = df_top20.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Top-20 Blackspots CSV",
        data=csv_top20,
        file_name="KMDA_Top20_Priority_Blackspots.csv",
        mime="text/csv"
    )

with c2:
    csv_actions = df_actions.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Prescribed Actions Tender List CSV",
        data=csv_actions,
        file_name="KMDA_Prescribed_Engineering_Actions.csv",
        mime="text/csv"
    )
