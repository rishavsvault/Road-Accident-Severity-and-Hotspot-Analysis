"""Page 3: Interactive What-If Counterfactual Policy Simulator.

Baseline is drawn from the worst crash on the selected blackspot segment,
not a global mean, for meaningful counterfactual comparison.
"""
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
import numpy as np
import plotly.graph_objects as go
import importlib
from src.models.xgboost_trainer import SeverityXGBoost

# Force reload of intervention engine in case Streamlit has an older version in memory
try:
    import src.explainability.intervention_engine as ie_mod
    importlib.reload(ie_mod)
    from src.explainability.intervention_engine import KMDAInterventionEngine
except Exception:
    from src.explainability.intervention_engine import KMDAInterventionEngine

try:
    from dashboard.components.disclaimer import render_disclaimer_banner, render_sidebar_disclaimer
except ImportError:
    from components.disclaimer import render_disclaimer_banner, render_sidebar_disclaimer

st.set_page_config(page_title="What-If Simulator | KMDA", layout="wide")

st.title("What-If Policy & Infrastructure Simulator")
render_disclaimer_banner("simulator")
render_sidebar_disclaimer()
st.markdown(
    "Simulate how targeted engineering interventions (street lighting, speed limits, medians) "
    "alter crash severity probabilities for a **specific blackspot segment** in Kolkata."
)

model_path  = "data/processed/xgboost_severity_model.joblib"
crashes_path = "data/processed/crash_level_features.parquet"
tiers_path  = "data/processed/segment_risk_tiers.csv"

if not os.path.exists(model_path) or not os.path.exists(crashes_path):
    st.error("Please run the pipeline first to train the model and save artifacts.")
    st.stop()

@st.cache_resource
def load_clf():
    import importlib
    import src.models.xgboost_trainer as xt
    importlib.reload(xt)
    clf_instance = xt.SeverityXGBoost.load_model(model_path)
    return clf_instance

@st.cache_data(ttl=600)
def load_data():
    crashes = pd.read_parquet(crashes_path)
    tiers = pd.read_csv(tiers_path) if os.path.exists(tiers_path) else pd.DataFrame()
    return crashes, tiers

clf = load_clf()

# Ensure predict_proba_calibrated exists even if an older cached instance is returned
if not hasattr(clf, "predict_proba_calibrated"):
    def _predict_proba_calibrated(X):
        if getattr(clf, "calibrated_model", None) is not None:
            return clf.calibrated_model.predict_proba(X)
        if hasattr(clf, "model") and hasattr(clf.model, "predict_proba"):
            return clf.model.predict_proba(X)
        raise AttributeError("Classifier has no predict_proba method.")
    clf.predict_proba_calibrated = _predict_proba_calibrated

df_crashes, df_tiers = load_data()

# ── Step 1: Select blackspot segment ────────────────────────────────────────
st.subheader("1. Select a Blackspot Segment")
if not df_tiers.empty:
    df_tiers_sorted = df_tiers.sort_values("segment_risk_score", ascending=False)
    seg_options = df_tiers_sorted.apply(
        lambda r: f"{r['segment_id']}  |  {r['corridor_name']}  |  Risk: {r['segment_risk_score']} / 100  |  {r['priority_tier']}",
        axis=1
    ).tolist()
    seg_ids     = df_tiers_sorted["segment_id"].tolist()
    selected_idx = st.selectbox("Choose segment (sorted by risk, highest first):", range(len(seg_options)),
                                 format_func=lambda i: seg_options[i], index=0)
    selected_seg_id = seg_ids[selected_idx]
else:
    selected_seg_id = None

# Find worst crash on this segment to anchor the baseline
if selected_seg_id is not None:
    seg_crashes = df_crashes[df_crashes["segment_id"] == selected_seg_id]
else:
    seg_crashes = df_crashes

if len(seg_crashes) > 0:
    # Baseline = worst (most fatal) crash conditions on the segment
    worst_crash = seg_crashes.sort_values("severity_code", ascending=False).iloc[0]
    baseline_row = worst_crash[clf.feature_names].to_dict()
    st.info(
        f"**Baseline anchored to the worst crash on segment `{selected_seg_id}`** "
        f"(Severity: {worst_crash.get('severity_label', 'Fatal')}, "
        f"Speed: {worst_crash.get('speed_limit', worst_crash.get('feat_speed_limit', 60)):.0f} km/h). "
        "Sliders below simulate improvements from this real worst-case starting point."
    )
else:
    # Fallback: use 90th-percentile risk profile from global crash data
    baseline_row = df_crashes[clf.feature_names].quantile(0.90).to_dict()
    st.info("Baseline anchored to the **90th-percentile risk profile** (no specific crash data for this segment).")

# Baseline parameters using simple names with fallback to feat_* columns
init_speed = int(baseline_row.get("speed_limit", baseline_row.get("feat_speed_limit", 60)))
init_is_lit = int(baseline_row.get("is_lit", baseline_row.get("feat_is_lit", 0)))
init_lanes = int(baseline_row.get("lanes", baseline_row.get("feat_lanes", 4)))
init_has_median = int(baseline_row.get("has_median", baseline_row.get("feat_has_median", 0)))
init_is_night = int(baseline_row.get("is_night", baseline_row.get("feat_is_night", 1)))
init_is_monsoon = int(baseline_row.get("is_monsoon", baseline_row.get("feat_is_monsoon", 0)))
init_precipitation = float(round(baseline_row.get("precipitation", baseline_row.get("feat_precipitation", 0.0))))
init_pedestrian = int(baseline_row.get("pedestrian_involved", baseline_row.get("feat_pedestrian_involved", 0)))

# ── Step 2: Intervention sliders ────────────────────────────────────────────
st.subheader("2. Configure Engineering Intervention Levers")
c1, c2, c3, c4 = st.columns(4)

with c1:
    valid_speed = int(round(min(max(init_speed, 20), 80) / 10.0) * 10)
    speed_limit = st.slider(
        "Speed Limit (km/h)",
        min_value=20, max_value=80,
        value=valid_speed,
        step=10,
        key=f"sim_speed_{selected_seg_id}"
    )
    is_lit = st.selectbox(
        "Street Lighting",
        [("No / Dim Lighting", 0), ("High-Mast LED Lighting", 1)],
        format_func=lambda x: x[0],
        index=min(max(init_is_lit, 0), 1),
        key=f"sim_lit_{selected_seg_id}"
    )[1]

with c2:
    valid_lanes = int(round(min(max(init_lanes, 2), 8) / 2.0) * 2)
    lanes = st.slider(
        "Number of Lanes",
        min_value=2, max_value=8,
        value=valid_lanes,
        step=2,
        key=f"sim_lanes_{selected_seg_id}"
    )
    has_median = st.selectbox(
        "Physical Median Barrier",
        [("No Median Barrier", 0), ("Concrete Crash Barrier", 1)],
        format_func=lambda x: x[0],
        index=min(max(init_has_median, 0), 1),
        key=f"sim_median_{selected_seg_id}"
    )[1]

with c3:
    is_night = st.selectbox(
        "Time of Day",
        [("Daytime (6:00 AM - 6:00 PM)", 0), ("Night (6:00 PM - 6:00 AM)", 1)],
        format_func=lambda x: x[0],
        index=min(max(init_is_night, 0), 1),
        key=f"sim_night_{selected_seg_id}"
    )[1]
    is_monsoon = st.selectbox(
        "Season",
        [("Dry / Winter Season", 0), ("Monsoon Season", 1)],
        format_func=lambda x: x[0],
        index=min(max(init_is_monsoon, 0), 1),
        key=f"sim_monsoon_{selected_seg_id}"
    )[1]

with c4:
    valid_precip = float(min(max(init_precipitation, 0.0), 30.0))
    precipitation = st.slider(
        "Precipitation (mm/h)",
        min_value=0.0, max_value=30.0,
        value=valid_precip,
        step=1.0,
        key=f"sim_precip_{selected_seg_id}"
    )
    pedestrian = st.selectbox(
        "Vulnerable Road Users",
        [("Vehicles Only", 0), ("Pedestrians / Cyclists Present", 1)],
        format_func=lambda x: x[0],
        index=min(max(init_pedestrian, 0), 1),
        key=f"sim_ped_{selected_seg_id}"
    )[1]

# Build intervention row from baseline overridden with simple parameter values
simulated_row = baseline_row.copy()

# Simple parameter assignments
simulated_row["speed_limit"]        = float(speed_limit)
simulated_row["is_high_speed"]      = 1 if speed_limit >= 50 else 0
simulated_row["lanes"]              = float(lanes)
simulated_row["is_lit"]             = int(is_lit)
simulated_row["has_median"]         = int(has_median)
simulated_row["is_night"]           = int(is_night)
simulated_row["is_monsoon"]         = int(is_monsoon)
simulated_row["precipitation"]      = float(precipitation)
simulated_row["is_raining"]         = 1 if precipitation > 0.1 else 0
simulated_row["pedestrian_involved"] = int(pedestrian)

# Mirror to model feature columns (feat_*) for classifier compatibility
simulated_row["feat_speed_limit"]        = float(speed_limit)
simulated_row["feat_is_high_speed"]      = 1 if speed_limit >= 50 else 0
simulated_row["feat_lanes"]              = float(lanes)
simulated_row["feat_is_lit"]             = int(is_lit)
simulated_row["feat_has_median"]         = int(has_median)
simulated_row["feat_is_night"]           = int(is_night)
simulated_row["feat_is_monsoon"]         = int(is_monsoon)
simulated_row["feat_precipitation"]      = float(precipitation)
simulated_row["feat_is_raining"]         = 1 if precipitation > 0.1 else 0
simulated_row["feat_pedestrian_involved"] = int(pedestrian)
simulated_row["feat_kinetic_energy"] = (float(speed_limit) ** 2) / 1000.0
simulated_row["feat_vulnerable_high_speed"] = 1 if ((int(pedestrian) or int(simulated_row.get("feat_is_two_wheeler", 0))) and speed_limit >= 50) else 0
simulated_row["feat_unlit_night"] = 1 if (int(is_night) and not int(is_lit)) else 0
simulated_row["feat_wet_high_speed"] = 1 if (float(precipitation) > 0.1 and speed_limit >= 50) else 0

# Safely construct DataFrames strictly matching model feature columns
baseline_df  = pd.DataFrame([[float(baseline_row.get(c, 0.0)) for c in clf.feature_names]], columns=clf.feature_names)
simulated_df = pd.DataFrame([[float(simulated_row.get(c, 0.0)) for c in clf.feature_names]], columns=clf.feature_names)

# Use calibrated probabilities if available
def _get_probs(model_obj, input_df):
    vals = input_df.values.astype(float)
    if hasattr(model_obj, "predict_proba_calibrated"):
        return model_obj.predict_proba_calibrated(vals)[0]
    elif getattr(model_obj, "calibrated_model", None) is not None:
        return model_obj.calibrated_model.predict_proba(vals)[0]
    elif hasattr(model_obj, "model") and hasattr(model_obj.model, "predict_proba"):
        return model_obj.model.predict_proba(vals)[0]
    elif hasattr(model_obj, "predict_proba"):
        return model_obj.predict_proba(vals)[0]
    raise AttributeError("Classifier has no probability prediction method.")

p_baseline  = _get_probs(clf, baseline_df)
p_simulated = _get_probs(clf, simulated_df)

# ── Step 3: Result comparison ────────────────────────────────────────────────
st.markdown("---")
st.subheader("3. Predicted Outcome Comparison")

labels = ["Minor / Non-Injury", "Grievous Injury", "Fatal"]
res_col1, res_col2 = st.columns([1.4, 1.0])

with res_col1:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=[round(p * 100, 1) for p in p_baseline],
        name="Baseline (Segment Worst-Case)", marker_color="#EF4444"
    ))
    fig.add_trace(go.Bar(
        x=labels, y=[round(p * 100, 1) for p in p_simulated],
        name="Simulated Scenario With Upgrades", marker_color="#3B82F6"
    ))
    fig.update_layout(
        barmode="group", yaxis_title="Probability (%)",
        height=380, margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

with res_col2:
    fatal_diff    = (p_simulated[2] - p_baseline[2]) * 100
    grievous_diff = (p_simulated[1] - p_baseline[1]) * 100

    # Causal Countermeasure Impact Analysis (AASHTO HSM / FHWA CMFs)
    def _compute_causal_impact(b_dict, s_dict, p_base, p_sim):
        if hasattr(KMDAInterventionEngine, "estimate_causal_countermeasure_impact"):
            return KMDAInterventionEngine.estimate_causal_countermeasure_impact(b_dict, s_dict, p_base, p_sim)
        base_fatal = float(p_base[2])
        sim_fatal = float(p_sim[2])
        ml_delta = (sim_fatal - base_fatal) * 100.0
        cmf_mult = 1.0
        cites = []
        if int(b_dict.get("is_lit", 1)) == 0 and int(s_dict.get("is_lit", 1)) == 1 and int(b_dict.get("is_night", 0)) == 1:
            cmf_mult *= 0.72
            cites.append("AASHTO HSM Lighting CMF: 0.72 (-28% night fatal risk)")
        if int(b_dict.get("has_median", 0)) == 0 and int(s_dict.get("has_median", 0)) == 1:
            cmf_mult *= 0.65
            cites.append("FHWA Median Barrier CMF: 0.65 (-35% cross-median fatal risk)")
        old_v = float(b_dict.get("speed_limit", 50))
        new_v = float(s_dict.get("speed_limit", 50))
        if old_v > 0 and new_v < old_v:
            v_ratio = new_v / old_v
            v_cmf = float(np.clip(v_ratio ** 4.0, 0.40, 1.0))
            cmf_mult *= v_cmf
            cites.append(f"Elvik Speed Power Model: CMF {v_cmf:.2f} (from {int(old_v)} to {int(new_v)} km/h)")
        causal_fatal = base_fatal * cmf_mult
        causal_delta = (causal_fatal - base_fatal) * 100.0
        return {
            "ml_predicted_fatal_pct": round(sim_fatal * 100.0, 1),
            "ml_fatal_delta_pct": round(ml_delta, 1),
            "cmf_compound_multiplier": round(cmf_mult, 3),
            "causal_fatal_delta_pct": round(causal_delta, 1),
            "consensus_fatal_delta_pct": round(0.5 * ml_delta + 0.5 * causal_delta, 1),
            "cmf_citations": cites if cites else ["Baseline operational conditions maintained"]
        }

    causal_res = _compute_causal_impact(
        b_dict=baseline_row if isinstance(baseline_row, dict) else baseline_row.to_dict(),
        s_dict=simulated_row if isinstance(simulated_row, dict) else simulated_row.to_dict(),
        p_base=p_baseline,
        p_sim=p_simulated
    )

    st.markdown("### Risk Impact Summary")
    m1, m2 = st.columns(2)
    m1.metric("ML Model Fatal Δ", f"{fatal_diff:+.1f}%",
              delta=f"{fatal_diff:+.1f}%", delta_color="inverse")
    m2.metric("Causal CMF Fatal Δ", f"{causal_res['causal_fatal_delta_pct']:+.1f}%",
              delta=f"{causal_res['causal_fatal_delta_pct']:+.1f}%", delta_color="inverse")

    st.markdown(f"**Consensus Risk Delta:** `{causal_res['consensus_fatal_delta_pct']:+.1f}%` *(50% ML + 50% Causal CMF)*")

    if fatal_diff < -2 or causal_res['causal_fatal_delta_pct'] < -2:
        st.success(f"[REDUCED] **Fatal Risk Reduced.** Consensus reduction: {abs(causal_res['consensus_fatal_delta_pct']):.1f}%")
        st.markdown("**Causal Evidence Base (HSM / IRC):**")
        for cite in causal_res["cmf_citations"]:
            st.markdown(f"- *{cite}*")
    elif fatal_diff > 2:
        st.error(f"[INCREASED] **Fatal Crash Probability Increased by {fatal_diff:.1f}%** vs baseline.")
        st.markdown("**Warning:** High speeds + low visibility + no median sharply elevate fatal impact likelihood.")
    else:
        st.info(
            "Minimal change detected. Interventions may be redundant with existing "
            "conditions on this segment, or the baseline already reflects low fatal risk."
        )

    st.caption(
        f"Baseline Fatal P: **{p_baseline[2]*100:.1f}%** -> "
        f"Simulated Fatal P: **{p_simulated[2]*100:.1f}%** | "
        f"Calibrated XGBoost with AASHTO HSM Countermeasure Cleared Factors."
    )

