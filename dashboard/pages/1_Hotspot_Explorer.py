"""Page 1: Interactive Kolkata Hotspot & Road Segment Risk Explorer."""
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
import pydeck as pdk
import plotly.express as px

try:
    from dashboard.components.disclaimer import render_disclaimer_banner, render_sidebar_disclaimer
except ImportError:
    from components.disclaimer import render_disclaimer_banner, render_sidebar_disclaimer

try:
    from dashboard.components.ui_cards import inject_custom_css, render_infographic_kpi
except ImportError:
    from components.ui_cards import inject_custom_css, render_infographic_kpi

st.set_page_config(page_title="Hotspot Explorer | KMDA", layout="wide")
inject_custom_css()

st.title("Kolkata Road Segment Hotspot Explorer")
render_disclaimer_banner("hotspots")
render_sidebar_disclaimer()

st.markdown(
    "Visualizing 200–500m segmented road units color-coded by composite severity-weighted risk index "
    "with traffic and population exposure normalization (0–100 scale)."
)

tiers_path = "data/processed/segment_risk_tiers.csv"
if not os.path.exists(tiers_path):
    st.error("Please run the pipeline first to generate segment risk tiers.")
    st.stop()

@st.cache_data(ttl=600)
def load_segments():
    return pd.read_csv(tiers_path)

df = load_segments()

# Sidebar filters
st.sidebar.header("🔍 Filter Road Segments")
tier_options = ["All"] + sorted(df["priority_tier"].dropna().unique().tolist()) if "priority_tier" in df.columns else ["All"]
selected_tier = st.sidebar.selectbox("Priority Tier", tier_options)

zone_options = ["All"] + sorted(df["spatial_zone_id"].dropna().unique().tolist()) if "spatial_zone_id" in df.columns else ["All"]
selected_zone = st.sidebar.selectbox("KMDA Planning Zone", zone_options)

corridor_options = ["All"] + sorted(df["corridor_name"].dropna().unique().tolist()) if "corridor_name" in df.columns else ["All"]
selected_corridor = st.sidebar.selectbox("Corridor Filter", corridor_options)
show_zero = st.sidebar.checkbox("Include unmonitored segments (0 crashes)", value=True)

# Filter logic
filtered_df = df.copy()
if selected_tier != "All" and "priority_tier" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["priority_tier"] == selected_tier]
if selected_zone != "All" and "spatial_zone_id" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["spatial_zone_id"] == selected_zone]
if selected_corridor != "All" and "corridor_name" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["corridor_name"] == selected_corridor]
if not show_zero and "total_crashes" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["total_crashes"] > 0]

# Dynamic Infographic Summary KPIs
f_segs   = len(filtered_df)
f_high   = int((filtered_df["priority_tier"] == "High Priority").sum()) if "priority_tier" in filtered_df.columns else 0
f_fatal  = int(filtered_df["fatal_crashes"].sum()) if "fatal_crashes" in filtered_df.columns else 0
f_mvkt   = float(filtered_df["two_year_mvkt"].sum()) if "two_year_mvkt" in filtered_df.columns else 0.0

hk1, hk2, hk3, hk4 = st.columns(4)
with hk1:
    render_infographic_kpi(
        title="Segments in View",
        value=f"{f_segs:,}",
        subtitle=f"Of {len(df)} mapped segments",
        icon="🛣️",
        color="blue",
        delta_badge="Filtered"
    )
with hk2:
    render_infographic_kpi(
        title="High-Priority Blackspots",
        value=f"{f_high:,}",
        subtitle="Requiring immediate civil work",
        icon="🚨",
        color="red",
        delta_badge="Critical"
    )
with hk3:
    render_infographic_kpi(
        title="Fatalities Recorded",
        value=f"{f_fatal:,}",
        subtitle="On filtered road stretches",
        icon="⚠️",
        color="amber",
        delta_badge="Life Hazard"
    )
with hk4:
    render_infographic_kpi(
        title="2-Year Traffic Exposure",
        value=f"{f_mvkt:,.1f} M-km",
        subtitle="Vehicle travel exposure (MVKT)",
        icon="🚗",
        color="green",
        delta_badge="Exposure Base"
    )

st.markdown('<div class="kmda-divider"></div>', unsafe_allow_html=True)

# Color mapping — zero-crash segments shown in grey to distinguish from true Low Priority
def get_color(row):
    if row.get("total_crashes", 0) == 0:
        return [148, 163, 184, 110]       # Grey / Slate — unmonitored / zero data
    if row.get("priority_tier") == "High Priority":
        return [220, 38, 38, 230]         # Red
    elif row.get("priority_tier") == "Medium Priority":
        return [245, 158, 11, 210]        # Orange
    else:
        return [59, 130, 246, 170]        # Blue

filtered_df = filtered_df.copy()
filtered_df["color"]  = filtered_df.apply(get_color, axis=1).tolist()
filtered_df["radius"] = filtered_df["segment_risk_score"].clip(5, 100) * 6

view_state = pdk.ViewState(latitude=22.5600, longitude=88.3800, zoom=11.2, pitch=35)

line_layer = pdk.Layer(
    "LineLayer",
    data=filtered_df,
    get_source_position="[start_lon, start_lat]",
    get_target_position="[end_lon, end_lat]",
    get_color="color",
    get_width=6,
    pickable=True
)
point_layer = pdk.Layer(
    "ScatterplotLayer",
    data=filtered_df,
    get_position="[mid_lon, mid_lat]",
    get_color="color",
    get_radius="radius",
    pickable=True,
    auto_highlight=True
)

deck = pdk.Deck(
    layers=[line_layer, point_layer],
    initial_view_state=view_state,
    tooltip={
        "html": """
            <b>Segment:</b> {segment_id}<br/>
            <b>Corridor:</b> {corridor_name}<br/>
            <b>Priority:</b> {priority_tier}<br/>
            <b>Risk Score:</b> {segment_risk_score} / 100<br/>
            <b>Total Crashes:</b> {total_crashes} (Fatal: {fatal_crashes}, Grievous: {grievous_crashes})<br/>
            <b>Est. AADT:</b> {estimated_aadt} veh/day | <b>2-Yr MVKT:</b> {two_year_mvkt} M-km<br/>
            <b>Speed Limit:</b> {speed_limit} km/h | <b>Lanes:</b> {lanes}<br/>
            <b>Note:</b> Grey segments = no crash record (unmonitored, not certified safe)
        """,
        "style": {"backgroundColor": "#0F172A", "color": "#F8FAFC", "fontSize": "13px", "padding": "10px"}
    },
    map_style="light"
)

# Render Pydeck 3D Map
st.pydeck_chart(deck, use_container_width=True)

# Visual Legend Bar
st.markdown("""
<div class="kmda-box" style="display: flex; gap: 14px; flex-wrap: wrap; align-items: center; justify-content: center; padding: 10px 16px; margin-top: 10px; margin-bottom: 20px;">
    <span style="font-size: 0.82rem; font-weight: 700; color: var(--kmda-text-muted);">MAP LEGEND:</span>
    <span class="kmda-badge badge-red">🔴 High Priority (Top 15% Risk - Action Needed)</span>
    <span class="kmda-badge badge-amber">🟠 Medium Priority (Preventive Action)</span>
    <span class="kmda-badge badge-blue">🔵 Low Priority (Baseline Monitoring)</span>
    <span class="kmda-badge badge-slate">⚪ Unmonitored (0 Crashes Logged)</span>
</div>
""", unsafe_allow_html=True)

# ── Infographic Corridor Leaderboard ───────────────────────────────────────
if len(filtered_df) > 0 and "segment_risk_score" in filtered_df.columns:
    st.markdown('<div class="kmda-divider"></div>', unsafe_allow_html=True)
    st.subheader("📊 Top 10 Blackspot Segments in Selected Filter")
    st.caption("Visual ranking of segments sorted by composite risk index (0–100 scale).")

    top10_filtered = filtered_df.sort_values("segment_risk_score", ascending=False).head(10).copy()
    top10_filtered["label"] = top10_filtered["segment_id"] + " (" + top10_filtered["corridor_name"].str[:22] + ")"

    fig_bar = px.bar(
        top10_filtered.sort_values("segment_risk_score", ascending=True),
        x="segment_risk_score",
        y="label",
        orientation="h",
        color="priority_tier",
        color_discrete_map={
            "High Priority": "#DC2626",
            "Medium Priority": "#F59E0B",
            "Low Priority / Baseline": "#3B82F6",
            "Unmonitored (No Data)": "#94A3B8"
        },
        text=top10_filtered.sort_values("segment_risk_score", ascending=True)["segment_risk_score"].apply(lambda v: f"{v:.1f}"),
        labels={"segment_risk_score": "Composite Risk Score (0-100)", "label": "Segment ID & Corridor"}
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(
        xaxis=dict(range=[0, 105]),
        yaxis=dict(title=""),
        height=360,
        margin=dict(l=15, r=25, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Inter, sans-serif")
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown('<div class="kmda-divider"></div>', unsafe_allow_html=True)

# ── Interactive Filterable Segment Table ────────────────────────────────────
st.subheader("📋 Segment Risk Ranks & Safety Improvement Metrics")
st.caption("Derived from AASHTO Highway Safety Manual (HSM) Safety Performance Functions & Empirical Bayes (EB) smoothing.")
display_cols = [c for c in [
    "rank", "segment_id", "corridor_name", "spatial_zone_id",
    "priority_tier", "segment_risk_score", "total_crashes",
    "fatal_crashes", "crash_rate_per_mvkt", "eb_expected_crashes",
    "potential_for_safety_improvement", "estimated_aadt", "speed_limit"
] if c in filtered_df.columns]
st.dataframe(filtered_df[display_cols].head(60), use_container_width=True)

# Spatial Autocorrelation Diagnostics (Moran's I)
moran_path = "data/processed/spatial_autocorrelation_moran.json"
if os.path.exists(moran_path):
    import json
    with open(moran_path, "r") as f:
        moran = json.load(f)
    with st.expander("🔬 Spatial Autocorrelation Diagnostics (Global Moran's I)"):
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Global Moran's I", f"{moran.get('morans_i', 0.0):.4f}")
        mc2.metric("Spatial Z-Score", f"{moran.get('z_score', 0.0):.2f}")
        mc3.metric("p-value", f"{moran.get('p_value', 1.0):.4f}")
        st.info(f"**Interpretation:** {moran.get('interpretation', 'N/A')}")

