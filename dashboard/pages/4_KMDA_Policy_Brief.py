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
import plotly.express as px

try:
    from dashboard.components.disclaimer import render_disclaimer_banner, render_sidebar_disclaimer
except ImportError:
    from components.disclaimer import render_disclaimer_banner, render_sidebar_disclaimer

try:
    from dashboard.components.ui_cards import inject_custom_css, render_infographic_kpi
except ImportError:
    from components.ui_cards import inject_custom_css, render_infographic_kpi

st.set_page_config(page_title="KMDA Policy Brief", layout="wide")
inject_custom_css()

# Official Executive Header
st.markdown("""
<div style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); padding: 24px 28px; border-radius: 12px; margin-bottom: 20px; color: #FFFFFF;">
    <div style="font-size: 0.80rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #38BDF8; margin-bottom: 4px;">
        Kolkata Metropolitan Development Authority • Directorate of Infrastructure
    </div>
    <div style="font-size: 1.85rem; font-weight: 800; font-family: 'Outfit', sans-serif; line-height: 1.2; margin-bottom: 6px;">
        Executive Policy Brief: Priority Road Safety Civil Interventions
    </div>
    <div style="font-size: 0.88rem; color: #94A3B8;">
        <b>To:</b> Municipal Commissioner & Chief Engineer | <b>Subject:</b> Ranked Blackspot Civil Packages | <b>Status:</b> Tender Pipeline Prototype
    </div>
</div>
""", unsafe_allow_html=True)

render_disclaimer_banner("policy")
render_sidebar_disclaimer()

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

# ── Top Infographic KPI Row ────────────────────────────────────────────────
tot_blackspots = len(df_top20)
tot_actions    = len(df_actions)
high_cost_cnt  = int((df_actions["estimated_cost_tier"] == "High").sum()) if "estimated_cost_tier" in df_actions.columns else 0
med_low_cnt    = tot_actions - high_cost_cnt

pk1, pk2, pk3, pk4 = st.columns(4)
with pk1:
    render_infographic_kpi(
        title="Priority Blackspots",
        value=f"{tot_blackspots}",
        subtitle="Ranked via NetKDE & HSM",
        icon="🎯",
        color="red",
        delta_badge="Top 20 Critical"
    )
with pk2:
    render_infographic_kpi(
        title="Prescribed Civil Items",
        value=f"{tot_actions}",
        subtitle="Across all city blackspots",
        icon="🛠️",
        color="blue",
        delta_badge="Tender Pipeline"
    )
with pk3:
    render_infographic_kpi(
        title="High-Cost Capital Works",
        value=f"{high_cost_cnt}",
        subtitle="Foot overbridges & flyovers",
        icon="🏗️",
        color="purple",
        delta_badge="Major Projects"
    )
with pk4:
    render_infographic_kpi(
        title="Fast-Track Safety Works",
        value=f"{med_low_cnt}",
        subtitle="High-mast LED, barriers & radar",
        icon="⚡",
        color="green",
        delta_badge="Immediate Term"
    )

st.markdown('<div class="kmda-divider"></div>', unsafe_allow_html=True)

# ── Infographic Visual Row: Resource Allocation & Agency Responsibilities ──
st.subheader("📊 Civil Budget Tier & Inter-Agency Implementation Breakdown")
st.caption("Distribution of engineering countermeasures across expenditure categories and municipal divisions.")

res_col1, res_col2 = st.columns([1.1, 1.4])

with res_col1:
    if "estimated_cost_tier" in df_actions.columns:
        cost_counts = df_actions["estimated_cost_tier"].value_counts().reset_index()
        cost_counts.columns = ["Cost Tier", "Interventions"]
        fig_cost = px.pie(
            cost_counts, names="Cost Tier", values="Interventions",
            color="Cost Tier",
            color_discrete_map={
                "High": "#DC2626",
                "Medium-High": "#F97316",
                "Medium": "#F59E0B",
                "Low": "#10B981"
            },
            hole=0.55
        )
        fig_cost.update_layout(
            title="Civil Expenditure Tiers",
            margin=dict(l=10, r=10, t=35, b=10),
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(text=f"<b>{tot_actions}</b><br><span style='font-size:11px;color:var(--kmda-text-muted);'>Items</span>",
                              x=0.5, y=0.5, font_size=16, showarrow=False)],
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            font=dict(family="Inter, sans-serif")
        )
        st.plotly_chart(fig_cost, use_container_width=True)

with res_col2:
    if "authority" in df_actions.columns:
        auth_counts = df_actions["authority"].value_counts().reset_index()
        auth_counts.columns = ["Executing Agency", "Action Count"]
        fig_auth = px.bar(
            auth_counts.sort_values("Action Count", ascending=True),
            x="Action Count", y="Executing Agency", orientation="h",
            text=auth_counts.sort_values("Action Count", ascending=True)["Action Count"],
            color="Action Count",
            color_continuous_scale=["#3B82F6", "#1E40AF"]
        )
        fig_auth.update_traces(textposition="outside")
        fig_auth.update_layout(
            title="Action Items by Executing Directorate",
            xaxis=dict(title="Tender Actions Count"),
            yaxis=dict(title=""),
            coloraxis_showscale=False,
            margin=dict(l=10, r=25, t=35, b=10),
            height=300,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif")
        )
        st.plotly_chart(fig_auth, use_container_width=True)

st.markdown('<div class="kmda-divider"></div>', unsafe_allow_html=True)

# ── Showcase Cards: Top 5 Critical Road Corridors ──────────────────────────
st.subheader("🔥 Top 5 Critical Corridors Requiring Priority Civil Tenders")
st.caption("Selected from NetKDE network density peaks and Empirical Bayes safety improvement potential.")

top5 = df_top20.head(5)
c_grid = st.columns(min(len(top5), 5))

for idx, (_, row) in enumerate(top5.iterrows()):
    with c_grid[idx]:
        st.markdown(f"""
        <div class="kmda-showcase-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span class="kmda-badge badge-red">Rank #{int(row.get('rank', idx+1))}</span>
                <span style="font-size: 0.72rem; font-weight: 700; color: var(--kmda-text-muted);">{row.get('spatial_zone_id', 'KMDA')}</span>
            </div>
            <div style="font-weight: 800; font-family: 'Outfit', sans-serif; font-size: 1.0rem; color: var(--kmda-text-primary); margin-bottom: 8px; line-height: 1.2;">
                {str(row.get('corridor_name', 'Corridor'))[:24]}
            </div>
            <div style="font-size: 0.78rem; color: var(--kmda-text-muted); margin-bottom: 6px;">
                <b>Risk Score:</b> <span style="font-weight: 700; color: #DC2626;">{row.get('segment_risk_score', 0):.1f} / 100</span>
            </div>
            <div style="font-size: 0.78rem; color: var(--kmda-text-muted); margin-bottom: 6px;">
                <b>Fatal Crashes:</b> {int(row.get('fatal_crashes', 0))} | Total: {int(row.get('total_crashes', 0))}
            </div>
            <div class="kmda-pill-tag">
                <b>Profile:</b> {int(row.get('speed_limit', 50))} km/h • {int(row.get('lanes', 4))} Lanes
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="kmda-divider"></div>', unsafe_allow_html=True)

from src.utils.feature_names import to_display_name

# ── AI Model Risk Factor Rankings ──────────────────────────────────────────
if not df_shap.empty:
    st.subheader("🧠 City-Wide Risk Drivers (TreeSHAP Severity Weights)")
    df_shap_display = df_shap.head(6).copy()
    df_shap_display["feature"] = df_shap_display["feature"].map(to_display_name).fillna(df_shap_display["feature"])

    fig_shap = px.bar(
        df_shap_display.sort_values("fatal_shap_importance", ascending=True),
        x="fatal_shap_importance", y="feature", orientation="h",
        text=df_shap_display.sort_values("fatal_shap_importance", ascending=True)["fatal_shap_importance"].apply(lambda v: f"{v:.3f}"),
        color="fatal_shap_importance",
        color_continuous_scale=["#F59E0B", "#DC2626"],
        labels={"fatal_shap_importance": "Mean |SHAP| Severity Weight", "feature": "Hazard Lever"}
    )
    fig_shap.update_traces(textposition="outside")
    fig_shap.update_layout(
        height=260,
        margin=dict(l=10, r=25, t=10, b=10),
        coloraxis_showscale=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif")
    )
    st.plotly_chart(fig_shap, use_container_width=True)

st.markdown('<div class="kmda-divider"></div>', unsafe_allow_html=True)

# ── Data Matrices & Download Actions ───────────────────────────────────────
st.subheader("📋 Top-20 Priority Blackspot Stretches Data Matrix")
display_seg_cols = [c for c in [
    "rank", "segment_id", "corridor_name", "spatial_zone_id",
    "segment_risk_score", "crash_rate_per_km", "total_crashes", "fatal_crashes",
    "grievous_crashes", "speed_limit", "lanes"
] if c in df_top20.columns]
st.dataframe(df_top20[display_seg_cols], use_container_width=True)

st.subheader("🛠️ Prescribed Municipal Engineering Tender Items")
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
        label="📥 Download Top-20 Blackspots CSV",
        data=csv_top20,
        file_name="KMDA_Top20_Priority_Blackspots.csv",
        mime="text/csv"
    )

with c2:
    csv_actions = df_actions.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Prescribed Actions Tender List CSV",
        data=csv_actions,
        file_name="KMDA_Prescribed_Engineering_Actions.csv",
        mime="text/csv"
    )
