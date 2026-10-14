"""Page 1: Interactive Kolkata Hotspot & Road Segment Risk Explorer."""
import streamlit as st
import pandas as pd
import pydeck as pdk
import os

st.set_page_config(page_title="Hotspot Explorer | KMDA", page_icon="🗺️", layout="wide")

st.title("🗺️ Kolkata Road Segment Hotspot Explorer")
st.markdown("Visualizing 200–500m segmented road units color-coded by composite severity-weighted risk index.")

tiers_path = "data/processed/segment_risk_tiers.csv"
if not os.path.exists(tiers_path):
    st.error("Please run the pipeline first to generate segment risk tiers.")
    st.stop()

df = pd.read_csv(tiers_path)

# Sidebar filters
st.sidebar.header("Filter Segments")
tier_options = ["All"] + list(df["priority_tier"].unique())
selected_tier = st.sidebar.selectbox("Priority Tier", tier_options)

zone_options = ["All"] + list(df["spatial_zone_id"].unique())
selected_zone = st.sidebar.selectbox("Planning Zone", zone_options)

corridor_options = ["All"] + list(df["corridor_name"].unique())
selected_corridor = st.sidebar.selectbox("Corridor", corridor_options)

# Filter logic
filtered_df = df.copy()
if selected_tier != "All":
    filtered_df = filtered_df[filtered_df["priority_tier"] == selected_tier]
if selected_zone != "All":
    filtered_df = filtered_df[filtered_df["spatial_zone_id"] == selected_zone]
if selected_corridor != "All":
    filtered_df = filtered_df[filtered_df["corridor_name"] == selected_corridor]

st.markdown(f"**Displaying {len(filtered_df)} of {len(df)} road segments.**")

# Color mapping by priority tier
def get_color(tier):
    if tier == "High Priority":
        return [220, 38, 38, 220]  # Red
    elif tier == "Medium Priority":
        return [245, 158, 11, 200]  # Orange
    else:
        return [59, 130, 246, 160]  # Blue

filtered_df["color"] = filtered_df["priority_tier"].apply(get_color)
filtered_df["radius"] = filtered_df["segment_risk_score"].clip(5, 100) * 6

# Pydeck Map Layer
view_state = pdk.ViewState(
    latitude=22.5600,
    longitude=88.3800,
    zoom=11.2,
    pitch=35
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

line_layer = pdk.Layer(
    "LineLayer",
    data=filtered_df,
    get_source_position="[start_lon, start_lat]",
    get_target_position="[end_lon, end_lat]",
    get_color="color",
    get_width=6,
    pickable=True
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
            <b>Speed Limit:</b> {speed_limit} km/h | <b>Lanes:</b> {lanes}
        """,
        "style": {"backgroundColor": "#0F172A", "color": "#F8FAFC", "fontSize": "13px", "padding": "10px"}
    },
    map_style="light"
)

st.pydeck_chart(deck, use_container_width=True)

# Data table summary
st.subheader("📋 Segment Risk Ranks")
st.dataframe(
    filtered_df[[
        "rank", "segment_id", "corridor_name", "spatial_zone_id",
        "priority_tier", "segment_risk_score", "total_crashes", "fatal_crashes", "speed_limit"
    ]].head(50),
    use_container_width=True
)
