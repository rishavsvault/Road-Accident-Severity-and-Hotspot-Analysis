"""Page 3: Interactive What-If Counterfactual Policy Simulator."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from src.models.xgboost_trainer import SeverityXGBoost

st.set_page_config(page_title="What-If Simulator | KMDA", page_icon="🛠️", layout="wide")

st.title("🛠️ What-If Policy & Infrastructure Simulator")
st.markdown("Simulate how municipal engineering interventions (lighting, speed limits, medians) alter fatal crash probabilities.")

model_path = "data/processed/xgboost_severity_model.joblib"
crashes_path = "data/processed/crash_level_features.parquet"

if not os.path.exists(model_path) or not os.path.exists(crashes_path):
    st.error("Please run the pipeline first to train the model and save artifacts.")
    st.stop()

clf = SeverityXGBoost.load_model(model_path)
df_crashes = pd.read_parquet(crashes_path)

st.subheader("1. Configure Environmental & Infrastructure Levers")

c1, c2, c3, c4 = st.columns(4)

with c1:
    speed_limit = st.slider("Speed Limit (km/h)", min_value=20, max_value=80, value=60, step=10)
    is_lit = st.selectbox("Street Lighting Installed?", [("Yes / High-Mast LED", 1), ("No / Dim / Non-functional", 0)], format_func=lambda x: x[0])[1]

with c2:
    lanes = st.slider("Number of Lanes", min_value=2, max_value=8, value=4, step=2)
    has_median = st.selectbox("Physical Median Barrier?", [("Yes / Concrete Barrier", 1), ("No Median", 0)], format_func=lambda x: x[0])[1]

with c3:
    is_night = st.selectbox("Time of Day", [("Night (22:00 - 05:00)", 1), ("Daytime (06:00 - 21:00)", 0)], format_func=lambda x: x[0])[1]
    is_monsoon = st.selectbox("Season", [("Monsoon Season", 1), ("Dry / Winter Season", 0)], format_func=lambda x: x[0])[1]

with c4:
    precipitation = st.slider("Precipitation (mm/h)", min_value=0.0, max_value=30.0, value=8.0 if is_monsoon else 0.0, step=1.0)
    pedestrian = st.selectbox("Vulnerable Road Users", [("Pedestrians / Cyclists Present", 1), ("Vehicles Only", 0)], format_func=lambda x: x[0])[1]

# Base input vector template
sample_row = df_crashes[clf.feature_names].mean().to_dict()

# Update with user choices
sample_row["feat_speed_limit"] = float(speed_limit)
sample_row["feat_is_high_speed"] = 1 if speed_limit >= 50 else 0
sample_row["feat_lanes"] = float(lanes)
sample_row["feat_is_lit"] = int(is_lit)
sample_row["feat_has_median"] = int(has_median)
sample_row["feat_is_night"] = int(is_night)
sample_row["feat_is_monsoon"] = int(is_monsoon)
sample_row["feat_precipitation"] = float(precipitation)
sample_row["feat_is_raining"] = 1 if precipitation > 0.1 else 0
sample_row["feat_pedestrian_involved"] = int(pedestrian)

# Baseline "Current" state (e.g. unmitigated)
current_row = sample_row.copy()
current_row["feat_is_lit"] = 0
current_row["feat_speed_limit"] = max(speed_limit, 60.0)
current_row["feat_has_median"] = 0

current_df = pd.DataFrame([current_row])[clf.feature_names]
simulated_df = pd.DataFrame([sample_row])[clf.feature_names]

p_current = clf.model.predict_proba(current_df.values)[0]
p_simulated = clf.model.predict_proba(simulated_df.values)[0]

st.markdown("---")
st.subheader("2. Predicted Outcome Comparison")

res_col1, res_col2 = st.columns([1.2, 1.0])

labels = ["Minor / Non-Injury", "Grievous Injury", "Fatal"]

with res_col1:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=[round(p * 100, 1) for p in p_current],
        name="Unmitigated Corridor State",
        marker_color="#94A3B8"
    ))
    fig.add_trace(go.Bar(
        x=labels,
        y=[round(p * 100, 1) for p in p_simulated],
        name="Simulated Scenario With Upgrades",
        marker_color="#3B82F6"
    ))
    fig.update_layout(
        barmode="group",
        yaxis_title="Probability (%)",
        height=380,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

with res_col2:
    fatal_diff = (p_simulated[2] - p_current[2]) * 100
    st.markdown("### Risk Impact Metric")
    if fatal_diff < 0:
        st.success(f"🔻 **Fatal Crash Probability Reduced by {abs(fatal_diff):.1f}%**")
        st.markdown("""
        **Policy Insight:**
        - Street lighting installation significantly reduces nighttime reaction lag.
        - Speed limit reductions reduce vehicle kinetic impact energy exponentially ($E_k \\propto v^2$).
        - Median barriers eliminate head-on crossover collisions.
        """)
    else:
        st.error(f"🔺 **Fatal Crash Probability Increased by {fatal_diff:.1f}%**")
        st.markdown("**Warning:** High speeds combined with low visibility or lack of median separation sharply elevate fatal impact likelihood.")
