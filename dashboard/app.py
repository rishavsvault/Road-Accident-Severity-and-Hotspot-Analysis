"""Main Entrypoint for the KMDA Road Safety & Hotspot Intelligence Dashboard."""
import streamlit as st
import pandas as pd
import os
import plotly.express as px

st.set_page_config(
    page_title="KMDA Road Safety Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🚦 KMDA Explainable Road Accident & Hotspot Intelligence System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Spatial Risk Ranking, XGBoost Severity Analytics & SHAP-Driven Countermeasure Planning for Kolkata</div>', unsafe_allow_html=True)

# Check if pipeline output exists
tiers_path = "data/processed/segment_risk_tiers.csv"
crashes_path = "data/processed/crash_level_features.parquet"
actions_path = "data/processed/kmda_prescribed_actions.csv"

if not os.path.exists(tiers_path) or not os.path.exists(crashes_path):
    st.warning("⚠️ Processed pipeline data not detected. Run `python run_pipeline.py` in your terminal to initialize models and spatial layers.")
    if st.button("🚀 Run Pipeline Now from Dashboard"):
        with st.spinner("Executing end-to-end pipeline (data generation, spatial snapping, KDE, XGBoost, SHAP)..."):
            from run_pipeline import run
            run()
            st.success("Pipeline executed successfully! Reloading data...")
            st.rerun()
    st.stop()

# Load summary data
df_segments = pd.read_csv(tiers_path)
df_crashes = pd.read_parquet(crashes_path)
df_actions = pd.read_csv(actions_path) if os.path.exists(actions_path) else pd.DataFrame()

# Top KPI row
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Crash Events Analyzed", f"{len(df_crashes):,}", delta="2-Year Multi-Zone Study")
with c2:
    fatal_count = (df_crashes["severity_code"] == 2).sum()
    st.metric("Fatal Crashes Identified", f"{fatal_count:,}", delta=f"{(fatal_count/len(df_crashes)*100):.1f}% of total", delta_color="inverse")
with c3:
    high_priority_segs = (df_segments["priority_tier"] == "High Priority").sum()
    st.metric("High-Priority Blackspot Segments", f"{high_priority_segs}", delta="KMDA Action Required", delta_color="inverse")
with c4:
    st.metric("Prescribed Engineering Fixes", f"{len(df_actions)}", delta="Ready for Tender")

st.markdown("---")

col_left, col_right = st.columns([1.6, 1.0])

with col_left:
    st.subheader("📊 Crash Distribution Across Kolkata Spatial Zones")
    zone_summary = df_crashes.groupby(["spatial_zone_id", "severity_label"]).size().reset_index(name="count")
    fig_zone = px.bar(
        zone_summary,
        x="spatial_zone_id",
        y="count",
        color="severity_label",
        color_discrete_map={
            "Fatal": "#EF4444",
            "Grievous Injury": "#F59E0B",
            "Minor / Non-Injury": "#10B981"
        },
        barmode="stack",
        labels={"spatial_zone_id": "Planning Zone", "count": "Crash Count", "severity_label": "Severity"}
    )
    fig_zone.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=350)
    st.plotly_chart(fig_zone, use_container_width=True)

with col_right:
    st.subheader("🎯 Priority Tier Allocation")
    tier_counts = df_segments["priority_tier"].value_counts().reset_index()
    tier_counts.columns = ["Priority Tier", "Segments"]
    fig_pie = px.pie(
        tier_counts,
        names="Priority Tier",
        values="Segments",
        color="Priority Tier",
        color_discrete_map={
            "High Priority": "#DC2626",
            "Medium Priority": "#F59E0B",
            "Low Priority / Baseline": "#3B82F6"
        },
        hole=0.45
    )
    fig_pie.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=350)
    st.plotly_chart(fig_pie, use_container_width=True)

st.info("💡 **Navigation:** Use the left sidebar to explore the interactive **Hotspot Map**, inspect **Segment SHAP Deep-Dives**, test counterfactuals in the **What-If Simulator**, or download the official **KMDA Policy Brief**.")
