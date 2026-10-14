"""Page 2: Segment-level SHAP Explainability & Infrastructure Breakdown."""
import streamlit as st
import pandas as pd
import plotly.express as px
import os
from src.models.xgboost_trainer import SeverityXGBoost
from src.explainability.shap_explainer import SeveritySHAPExplainer
from src.explainability.intervention_engine import KMDAInterventionEngine

st.set_page_config(page_title="Segment Deep Dive | KMDA", page_icon="🔍", layout="wide")

st.title("🔍 Segment Deep Dive & Explainable SHAP Analysis")
st.markdown("Inspect local SHAP drivers behind predicted crash severity on specific Kolkata road stretches.")

model_path = "data/processed/xgboost_severity_model.joblib"
tiers_path = "data/processed/segment_risk_tiers.csv"
crashes_path = "data/processed/crash_level_features.parquet"

if not os.path.exists(model_path) or not os.path.exists(tiers_path) or not os.path.exists(crashes_path):
    st.error("Please run the pipeline first to train the model and generate feature datasets.")
    st.stop()

# Load artifacts
df_segments = pd.read_csv(tiers_path)
df_crashes = pd.read_parquet(crashes_path)
clf = SeverityXGBoost.load_model(model_path)
shap_tool = SeveritySHAPExplainer(clf.model, clf.feature_names)
engine = KMDAInterventionEngine()

# Select high-risk segment
seg_list = df_segments["segment_id"].tolist()
selected_seg_id = st.selectbox("Select Segment ID to Analyze:", seg_list, index=0)

seg_data = df_segments[df_segments["segment_id"] == selected_seg_id].iloc[0]

# Display segment metadata cards
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Corridor", seg_data["corridor_name"])
col2.metric("Priority Tier", seg_data["priority_tier"])
col3.metric("Risk Score", f"{seg_data['segment_risk_score']}")
col4.metric("Total Crashes", f"{seg_data['total_crashes']}")
col5.metric("Fatal Crashes", f"{seg_data['fatal_crashes']}", delta_color="inverse")

st.markdown("---")

# Find crashes associated with this segment
seg_crashes = df_crashes[df_crashes["segment_id"] == selected_seg_id]

if len(seg_crashes) == 0:
    st.info("No recorded crash events on this specific segment stretch. Modeling baseline risk from road attributes.")
    sample_crash = df_crashes.iloc[0]
else:
    sample_crash = seg_crashes.iloc[0]

# Compute SHAP explanation for the instance
shap_drivers = shap_tool.explain_single_instance(sample_crash)
df_shap = pd.DataFrame(shap_drivers[:10])

c_left, c_right = st.columns([1.2, 1.0])

with c_left:
    st.subheader("🎯 Local SHAP Drivers: Risk Elevators vs Mitigators")
    fig = px.bar(
        df_shap,
        x="shap_value",
        y="feature",
        orientation="h",
        color="impact",
        color_discrete_map={
            "Increases Risk": "#EF4444",
            "Decreases Risk": "#10B981"
        },
        labels={"shap_value": "SHAP Contribution to Fatal Crash Risk", "feature": "Feature"},
        title="Top 10 Feature Contributions for this Segment"
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), height=420)
    st.plotly_chart(fig, use_container_width=True)

with c_right:
    st.subheader("🛠️ Prescribed KMDA Engineering Countermeasures")
    recommended = engine.recommend_for_segment(seg_data.to_dict(), shap_drivers)
    if not recommended:
        st.success("No immediate structural civil interventions triggered for this segment.")
    for rec in recommended:
        with st.container():
            st.markdown(f"**Action:** `{rec['recommended_action']}`")
            st.caption(f"Authority: {rec['authority']} | Priority: {rec['priority']} | Cost: {rec['estimated_cost_tier']}")
            st.markdown("---")
