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

try:
    from dashboard.components.disclaimer import render_disclaimer_banner, render_sidebar_disclaimer
except ImportError:
    from components.disclaimer import render_disclaimer_banner, render_sidebar_disclaimer

st.set_page_config(page_title="Hotspot Explorer | KMDA", layout="wide")

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
st.sidebar.header("Filter Segments")
tier_options = ["All"] + sorted(df["priority_tier"].dropna().unique().tolist()) if "priority_tier" in df.columns else ["All"]
selected_tier = st.sidebar.selectbox("Priority Tier", tier_options)

zone_options = ["All"] + sorted(df["spatial_zone_id"].dropna().unique().tolist()) if "spatial_zone_id" in df.columns else ["All"]
selected_zone = st.sidebar.selectbox("Planning Zone", zone_options)

corridor_options = ["All"] + sorted(df["corridor_name"].dropna().unique().tolist()) if "corridor_name" in df.columns else ["All"]
selected_corridor = st.sidebar.selectbox("Corridor", corridor_options)
show_zero = st.sidebar.checkbox("Show segments with zero crashes (Unmonitored)", value=True)

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

st.markdown(f"**Displaying {len(filtered_df)} of {len(df)} road segments.**")

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

st.pydeck_chart(deck, use_container_width=True)

# Legend
with st.expander("Map Legend & Technical Notes"):
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown("**High Priority (Red)**: Top 15% risk; immediate civil action required")
    col2.markdown("**Medium Priority (Orange)**: Preventive maintenance zone")
    col3.markdown("**Low Priority (Blue)**: Baseline monitoring only")
    col4.markdown("**Unmonitored (Grey)**: Zero crash records in study period (not certified safe)")

st.subheader("Segment Risk Ranks (AASHTO Highway Safety Manual & Empirical Bayes)")
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
    with st.expander("Spatial Autocorrelation Diagnostics (Global Moran's I)"):
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Global Moran's I", f"{moran.get('morans_i', 0.0):.4f}")
        mc2.metric("Spatial Z-Score", f"{moran.get('z_score', 0.0):.2f}")
        mc3.metric("p-value", f"{moran.get('p_value', 1.0):.4f}")
        st.info(f"**Interpretation:** {moran.get('interpretation', 'N/A')}")

