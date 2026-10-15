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

try:
    from dashboard.components.ui_cards import inject_custom_css, render_infographic_kpi
except ImportError:
    from components.ui_cards import inject_custom_css, render_infographic_kpi

st.set_page_config(
    page_title="KMDA Road Safety Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom design tokens & responsive CSS
inject_custom_css()

# Official Data Disclaimer Banner
render_disclaimer_banner("general")
render_sidebar_disclaimer()

# Branded Executive Header
st.markdown("""
<div style="background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%); padding: 26px 30px; border-radius: 14px; margin-bottom: 22px; color: #FFFFFF; box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
        <div>
            <div style="font-size: 0.80rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #38BDF8; margin-bottom: 4px;">
                Govt. of West Bengal • Urban Development & Municipal Affairs
            </div>
            <div style="font-size: 2.1rem; font-weight: 800; font-family: 'Outfit', sans-serif; line-height: 1.15; margin-bottom: 6px;">
                KMDA Road Safety & Hotspot Intelligence System
            </div>
            <div style="font-size: 1.0rem; color: #94A3B8; font-weight: 400; max-width: 820px;">
                Network-Constrained Spatial Risk Analytics (NetKDE), Calibrated XGBoost Severity Modeling & AASHTO Causal Interventions for Kolkata
            </div>
        </div>
        <div style="text-align: right;">
            <span class="kmda-badge badge-blue" style="font-size: 0.78rem; padding: 6px 12px;">Prototype Stage</span>
            <div style="font-size: 0.75rem; color: #64748B; margin-top: 6px;">Kolkata Police & KMDA Zones</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Quick Navigation Guide Banner
st.markdown("""
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-bottom: 22px;">
    <div class="kmda-box" style="padding: 12px 14px; border-radius: 10px;">
        <div style="font-weight: 700; font-size: 0.88rem; color: var(--kmda-text-primary);">🗺️ 1. Hotspot Explorer</div>
        <div style="font-size: 0.75rem; color: var(--kmda-text-muted);">Interactive 3D NetKDE segment map with filterable priority tiers.</div>
    </div>
    <div class="kmda-box" style="padding: 12px 14px; border-radius: 10px;">
        <div style="font-weight: 700; font-size: 0.88rem; color: var(--kmda-text-primary);">🔍 2. Segment Deep Dive</div>
        <div style="font-size: 0.75rem; color: var(--kmda-text-muted);">Explainable TreeSHAP feature drivers & physical cross-section analysis.</div>
    </div>
    <div class="kmda-box" style="padding: 12px 14px; border-radius: 10px;">
        <div style="font-weight: 700; font-size: 0.88rem; color: var(--kmda-text-primary);">⚡ 3. What-If Simulator</div>
        <div style="font-size: 0.75rem; color: var(--kmda-text-muted);">Counterfactual engineering simulator with AASHTO HSM causal CMFs.</div>
    </div>
    <div class="kmda-box" style="padding: 12px 14px; border-radius: 10px;">
        <div style="font-weight: 700; font-size: 0.88rem; color: var(--kmda-text-primary);">📋 4. KMDA Policy Brief</div>
        <div style="font-size: 0.75rem; color: var(--kmda-text-muted);">Top 20 blackspots list & tender-ready engineering action plan.</div>
    </div>
</div>
""", unsafe_allow_html=True)

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

# ── Top Infographic KPI Row ────────────────────────────────────────────────
fatal_count = int((df_crashes["severity_code"] == 2).sum())
fatal_pct   = (fatal_count / len(df_crashes) * 100) if len(df_crashes) > 0 else 0
high_priority_segs = int((df_segments["priority_tier"] == "High Priority").sum()) if "priority_tier" in df_segments.columns else 0

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    render_infographic_kpi(
        title="Crash Events Analyzed",
        value=f"{len(df_crashes):,}",
        subtitle="Across all Kolkata zones",
        icon="🚗",
        color="blue",
        delta_badge="2-Year Study"
    )
with kpi2:
    render_infographic_kpi(
        title="Fatal Crashes Identified",
        value=f"{fatal_count:,}",
        subtitle=f"{fatal_pct:.1f}% fatality share",
        icon="⚠️",
        color="red",
        delta_badge="Critical Risk"
    )
with kpi3:
    render_infographic_kpi(
        title="High-Priority Blackspots",
        value=f"{high_priority_segs}",
        subtitle="Urgent engineering attention",
        icon="🎯",
        color="amber",
        delta_badge="Top 15% Risk"
    )
with kpi4:
    render_infographic_kpi(
        title="Engineering Fixes Prescribed",
        value=f"{len(df_actions)}",
        subtitle="Actionable tender items",
        icon="🛠️",
        color="green",
        delta_badge="Tender Ready"
    )

st.markdown('<div class="kmda-divider"></div>', unsafe_allow_html=True)

# ── Infographic Visual Row 1: Spatial & Priority Allocations ───────────────
col_left, col_right = st.columns([1.5, 1.0])

with col_left:
    st.subheader("🗺️ Crash Severity Profile by Planning Zone")
    zone_summary = df_crashes.groupby(["spatial_zone_id", "severity_label"]).size().reset_index(name="count")
    fig_zone = px.bar(
        zone_summary, x="spatial_zone_id", y="count", color="severity_label",
        color_discrete_map={"Fatal": "#EF4444", "Grievous Injury": "#F59E0B", "Minor / Non-Injury": "#10B981"},
        barmode="stack",
        labels={"spatial_zone_id": "KMDA Planning Zone", "count": "Recorded Crashes", "severity_label": "Severity"}
    )
    fig_zone.update_layout(
        margin=dict(l=15, r=15, t=15, b=15),
        height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif")
    )
    st.plotly_chart(fig_zone, use_container_width=True)

with col_right:
    st.subheader("🎯 Priority Tier Allocation")
    tier_counts = df_segments["priority_tier"].value_counts().reset_index()
    tier_counts.columns = ["Priority Tier", "Segments"]
    total_segs = len(df_segments)
    fig_pie = px.pie(
        tier_counts, names="Priority Tier", values="Segments",
        color="Priority Tier",
        color_discrete_map={
            "High Priority": "#DC2626",
            "Medium Priority": "#F59E0B",
            "Low Priority / Baseline": "#3B82F6",
            "Unmonitored (No Data)": "#94A3B8"
        },
        hole=0.55
    )
    fig_pie.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=340,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(text=f"<b>{total_segs}</b><br><span style='font-size:11px;color:#94A3B8;'>Segments</span>",
                          x=0.5, y=0.5, font_size=17, showarrow=False)],
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        font=dict(family="Inter, sans-serif")
    )
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown('<div class="kmda-divider"></div>', unsafe_allow_html=True)

# ── Infographic Visual Row 2: Accident Drivers & Danger Corridors ──────────
col_factors, col_corridors = st.columns([1.1, 1.3])

with col_factors:
    st.subheader("📊 Key Accident Hazard Breakdown")
    st.caption("Proportion of crashes occurring under specific environmental and exposure hazards.")

    night_pct = float(df_crashes["feat_is_night"].mean() * 100) if "feat_is_night" in df_crashes.columns else 0.0
    monsoon_pct = float(df_crashes["feat_is_monsoon"].mean() * 100) if "feat_is_monsoon" in df_crashes.columns else 0.0
    ped_pct = float(df_crashes["feat_pedestrian_involved"].mean() * 100) if "feat_pedestrian_involved" in df_crashes.columns else 0.0
    high_speed_pct = float(df_crashes["feat_is_high_speed"].mean() * 100) if "feat_is_high_speed" in df_crashes.columns else 0.0

    factors_df = pd.DataFrame([
        {"Factor": "🌙 Nighttime (6 PM - 6 AM)", "Share": night_pct, "Color": "#6366F1"},
        {"Factor": "🌧️ Monsoon Season", "Share": monsoon_pct, "Color": "#0EA5E9"},
        {"Factor": "🚶 Pedestrian / VRU Exposure", "Share": ped_pct, "Color": "#EF4444"},
        {"Factor": "⚡ High Speed (>= 50 km/h)", "Share": high_speed_pct, "Color": "#F59E0B"}
    ]).sort_values("Share", ascending=True)

    fig_factors = px.bar(
        factors_df, x="Share", y="Factor", orientation="h",
        text=factors_df["Share"].apply(lambda v: f"{v:.1f}%"),
        color="Factor",
        color_discrete_sequence=["#6366F1", "#0EA5E9", "#F59E0B", "#EF4444"]
    )
    fig_factors.update_traces(textposition="outside", showlegend=False)
    fig_factors.update_layout(
        xaxis=dict(range=[0, 100], title="Percentage of Total Crashes (%)"),
        yaxis=dict(title=""),
        margin=dict(l=10, r=20, t=10, b=10),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif")
    )
    st.plotly_chart(fig_factors, use_container_width=True)

with col_corridors:
    st.subheader("🔥 Top 5 Hazardous Road Corridors")
    st.caption("Ranked by average composite segment risk score across monitored stretches.")

    if "corridor_name" in df_segments.columns and "segment_risk_score" in df_segments.columns:
        corr_summary = df_segments.groupby("corridor_name").agg(
            avg_risk=("segment_risk_score", "mean"),
            fatal_crashes=("fatal_crashes", "sum"),
            total_crashes=("total_crashes", "sum"),
            segments=("segment_id", "count")
        ).reset_index().sort_values("avg_risk", ascending=False).head(5)

        fig_corr = px.bar(
            corr_summary.sort_values("avg_risk", ascending=True),
            x="avg_risk", y="corridor_name", orientation="h",
            text=corr_summary.sort_values("avg_risk", ascending=True)["avg_risk"].apply(lambda v: f"{v:.1f} / 100"),
            color="avg_risk",
            color_continuous_scale=["#F59E0B", "#DC2626"],
            labels={"avg_risk": "Mean Segment Risk", "corridor_name": "Corridor"}
        )
        fig_corr.update_traces(textposition="outside")
        fig_corr.update_layout(
            xaxis=dict(range=[0, 105], title="Composite Risk Index (0-100)"),
            yaxis=dict(title=""),
            coloraxis_showscale=False,
            margin=dict(l=10, r=25, t=10, b=10),
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif")
        )
        st.plotly_chart(fig_corr, use_container_width=True)

st.markdown('<div class="kmda-divider"></div>', unsafe_allow_html=True)

# ── Production Model Validation & Calibration Section ───────────────────────
st.subheader("🛡️ AI Model Reliability & Cross-Validation Benchmarks")
st.caption("Evaluated with 5-Fold Spatial Grouped Cross-Validation (unseen geographic zones) and Isotonic Calibration.")

bench_col1, bench_col2 = st.columns([1.1, 1.3])

with bench_col1:
    if not df_compare.empty:
        prod_df = df_compare[df_compare["model"].str.contains("XGBoost", case=False, na=False)].copy()
        if prod_df.empty:
            prod_df = df_compare.head(1).copy()
        p_row = prod_df.iloc[0]

        st.markdown(f"""
        <div class="kmda-box" style="margin-bottom: 12px;">
            <div style="font-size: 0.82rem; font-weight: 700; color: var(--kmda-text-muted); text-transform: uppercase; margin-bottom: 8px;">
                Selected Architecture: {p_row.get('model', 'Calibrated Multiclass XGBoost')}
            </div>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
                <div class="kmda-inner-box" style="text-align: center;">
                    <div style="font-size: 0.72rem; color: var(--kmda-text-muted); text-transform: uppercase;">Macro F1-Score</div>
                    <div style="font-size: 1.3rem; font-weight: 800; color: var(--kmda-text-primary);">{float(p_row.get('macro_f1', 0.0)):.3f}</div>
                    <span class="kmda-badge badge-blue">Spatial CV</span>
                </div>
                <div class="kmda-inner-box" style="text-align: center;">
                    <div style="font-size: 0.72rem; color: var(--kmda-text-muted); text-transform: uppercase;">Fatal Recall</div>
                    <div style="font-size: 1.3rem; font-weight: 800; color: #DC2626;">{float(p_row.get('fatal_recall', 0.0)):.3f}</div>
                    <span class="kmda-badge badge-red">Life Safety</span>
                </div>
                <div class="kmda-inner-box" style="text-align: center;">
                    <div style="font-size: 0.72rem; color: var(--kmda-text-muted); text-transform: uppercase;">Fatal PR-AUC</div>
                    <div style="font-size: 1.3rem; font-weight: 800; color: #059669;">{float(p_row.get('pr_auc_fatal', 0.0)):.3f}</div>
                    <span class="kmda-badge badge-green">Imbalance Robust</span>
                </div>
                <div class="kmda-inner-box" style="text-align: center;">
                    <div style="font-size: 0.72rem; color: var(--kmda-text-muted); text-transform: uppercase;">Quadratic Kappa</div>
                    <div style="font-size: 1.3rem; font-weight: 800; color: #7C3AED;">{float(p_row.get('quadratic_weighted_kappa', 0.0)):.3f}</div>
                    <span class="kmda-badge badge-purple">Ordinal Fit</span>
                </div>
            </div>
            <div style="font-size: 0.74rem; color: var(--kmda-text-muted); margin-top: 12px; line-height: 1.4;">
                *Evaluated by holding out entire geographic planning zones to ensure the model generalizes to unmapped corridors.*
            </div>
        </div>
        """, unsafe_allow_html=True)

with bench_col2:
    if calib_data:
        fig_calib = go.Figure()
        fig_calib.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Perfect Calibration",
            line=dict(color="#94A3B8", dash="dash", width=1.5)
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
                    marker=dict(size=6)
                ))
        fig_calib.update_layout(
            title="Reliability Diagram (Predicted vs. Observed)",
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Empirical Frequency",
            height=280,
            margin=dict(l=15, r=15, t=35, b=15),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif")
        )
        st.plotly_chart(fig_calib, use_container_width=True)

st.markdown('<div class="kmda-divider"></div>', unsafe_allow_html=True)
st.caption("Kolkata Metropolitan Development Authority • Road Safety & Urban Mobility Division • Built for Evidence-Based Civil Tender Allocation")

if st.sidebar.button("Clear Cache & Reload Data"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()
