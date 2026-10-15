"""Reusable UI styling, adaptive theme tokens (Light/Dark/System mode), and infographic visualization components for the KMDA Dashboard."""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def inject_custom_css():
    """Injects high-end, theme-adaptive CSS tokens that seamlessly comply with both System Light & Dark modes."""
    st.markdown("""
    <style>
        /* Import Modern Google Typography */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Outfit:wght@500;600;700;800&display=swap');

        /* ── Base Theme Design Tokens (Default: Light Mode) ────────────────── */
        :root {
            --kmda-card-bg-gradient: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(248, 250, 252, 0.90));
            --kmda-card-border: rgba(226, 232, 240, 0.9);
            --kmda-card-shadow: 0 4px 15px -3px rgba(0, 0, 0, 0.05), 0 2px 6px -2px rgba(0, 0, 0, 0.03);
            --kmda-card-shadow-hover: 0 10px 22px -4px rgba(0, 0, 0, 0.08);
            --kmda-card-title: #64748B;
            --kmda-card-val: #0F172A;
            --kmda-card-sub: #64748B;

            --kmda-box-bg: rgba(248, 250, 252, 0.9);
            --kmda-box-border: #E2E8F0;
            --kmda-inner-bg: #FFFFFF;
            --kmda-inner-border: #E2E8F0;

            --kmda-text-primary: #0F172A;
            --kmda-text-secondary: #334155;
            --kmda-text-muted: #64748B;
            --kmda-divider-grad: linear-gradient(90deg, #E2E8F0 0%, #CBD5E1 50%, #E2E8F0 100%);

            /* Pill Badges */
            --badge-red-bg: #FEE2E2; --badge-red-text: #DC2626; --badge-red-border: #FECACA;
            --badge-amber-bg: #FEF3C7; --badge-amber-text: #D97706; --badge-amber-border: #FDE68A;
            --badge-blue-bg: #DBEAFE; --badge-blue-text: #2563EB; --badge-blue-border: #BFDBFE;
            --badge-green-bg: #D1FAE5; --badge-green-text: #059669; --badge-green-border: #A7F3D0;
            --badge-purple-bg: #EDE9FE; --badge-purple-text: #7C3AED; --badge-purple-border: #DDD6FE;
            --badge-slate-bg: #F1F5F9; --badge-slate-text: #475569; --badge-slate-border: #E2E8F0;
        }

        /* ── System Dark Mode Media Query & Streamlit Theme Hook ──────────── */
        @media (prefers-color-scheme: dark) {
            :root, .stApp {
                --kmda-card-bg-gradient: linear-gradient(135deg, rgba(30, 41, 59, 0.70), rgba(15, 23, 42, 0.85));
                --kmda-card-border: rgba(71, 85, 105, 0.55);
                --kmda-card-shadow: 0 4px 15px -3px rgba(0, 0, 0, 0.40);
                --kmda-card-shadow-hover: 0 10px 22px -4px rgba(0, 0, 0, 0.60);
                --kmda-card-title: #94A3B8;
                --kmda-card-val: #F8FAFC;
                --kmda-card-sub: #94A3B8;

                --kmda-box-bg: rgba(30, 41, 59, 0.55);
                --kmda-box-border: rgba(71, 85, 105, 0.45);
                --kmda-inner-bg: rgba(15, 23, 42, 0.75);
                --kmda-inner-border: rgba(51, 65, 85, 0.70);

                --kmda-text-primary: #F8FAFC;
                --kmda-text-secondary: #E2E8F0;
                --kmda-text-muted: #94A3B8;
                --kmda-divider-grad: linear-gradient(90deg, #334155 0%, #475569 50%, #334155 100%);

                /* Dark Pill Badges */
                --badge-red-bg: rgba(220, 38, 38, 0.20); --badge-red-text: #F87171; --badge-red-border: rgba(239, 68, 68, 0.35);
                --badge-amber-bg: rgba(217, 119, 6, 0.20); --badge-amber-text: #FBBF24; --badge-amber-border: rgba(245, 158, 11, 0.35);
                --badge-blue-bg: rgba(37, 99, 235, 0.20); --badge-blue-text: #60A5FA; --badge-blue-border: rgba(59, 130, 246, 0.35);
                --badge-green-bg: rgba(5, 150, 105, 0.20); --badge-green-text: #34D399; --badge-green-border: rgba(16, 185, 129, 0.35);
                --badge-purple-bg: rgba(124, 58, 237, 0.20); --badge-purple-text: #A78BFA; --badge-purple-border: rgba(139, 92, 246, 0.35);
                --badge-slate-bg: rgba(51, 65, 85, 0.45); --badge-slate-text: #CBD5E1; --badge-slate-border: rgba(71, 85, 105, 0.50);
            }
        }

        /* Streamlit Dark Theme Class Overrides */
        [data-theme="dark"],
        [data-base-theme="dark"],
        .stApp[data-theme="dark"],
        .stApp[data-base-theme="dark"],
        [data-testid="stAppViewContainer"]:has([data-theme="dark"]) {
            --kmda-card-bg-gradient: linear-gradient(135deg, rgba(30, 41, 59, 0.70), rgba(15, 23, 42, 0.85)) !important;
            --kmda-card-border: rgba(71, 85, 105, 0.55) !important;
            --kmda-card-shadow: 0 4px 15px -3px rgba(0, 0, 0, 0.40) !important;
            --kmda-card-shadow-hover: 0 10px 22px -4px rgba(0, 0, 0, 0.60) !important;
            --kmda-card-title: #94A3B8 !important;
            --kmda-card-val: #F8FAFC !important;
            --kmda-card-sub: #94A3B8 !important;

            --kmda-box-bg: rgba(30, 41, 59, 0.55) !important;
            --kmda-box-border: rgba(71, 85, 105, 0.45) !important;
            --kmda-inner-bg: rgba(15, 23, 42, 0.75) !important;
            --kmda-inner-border: rgba(51, 65, 85, 0.70) !important;

            --kmda-text-primary: #F8FAFC !important;
            --kmda-text-secondary: #E2E8F0 !important;
            --kmda-text-muted: #94A3B8 !important;
            --kmda-divider-grad: linear-gradient(90deg, #334155 0%, #475569 50%, #334155 100%) !important;

            --badge-red-bg: rgba(220, 38, 38, 0.20) !important; --badge-red-text: #F87171 !important; --badge-red-border: rgba(239, 68, 68, 0.35) !important;
            --badge-amber-bg: rgba(217, 119, 6, 0.20) !important; --badge-amber-text: #FBBF24 !important; --badge-amber-border: rgba(245, 158, 11, 0.35) !important;
            --badge-blue-bg: rgba(37, 99, 235, 0.20) !important; --badge-blue-text: #60A5FA !important; --badge-blue-border: rgba(59, 130, 246, 0.35) !important;
            --badge-green-bg: rgba(5, 150, 105, 0.20) !important; --badge-green-text: #34D399 !important; --badge-green-border: rgba(16, 185, 129, 0.35) !important;
            --badge-purple-bg: rgba(124, 58, 237, 0.20) !important; --badge-purple-text: #A78BFA !important; --badge-purple-border: rgba(139, 92, 246, 0.35) !important;
            --badge-slate-bg: rgba(51, 65, 85, 0.45) !important; --badge-slate-text: #CBD5E1 !important; --badge-slate-border: rgba(71, 85, 105, 0.50) !important;
        }

        /* Streamlit Light Theme Explicit Override */
        [data-theme="light"],
        [data-base-theme="light"],
        .stApp[data-theme="light"],
        .stApp[data-base-theme="light"] {
            --kmda-card-bg-gradient: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(248, 250, 252, 0.90)) !important;
            --kmda-card-border: rgba(226, 232, 240, 0.9) !important;
            --kmda-card-shadow: 0 4px 15px -3px rgba(0, 0, 0, 0.05), 0 2px 6px -2px rgba(0, 0, 0, 0.03) !important;
            --kmda-card-shadow-hover: 0 10px 22px -4px rgba(0, 0, 0, 0.08) !important;
            --kmda-card-title: #64748B !important;
            --kmda-card-val: #0F172A !important;
            --kmda-card-sub: #64748B !important;

            --kmda-box-bg: rgba(248, 250, 252, 0.9) !important;
            --kmda-box-border: #E2E8F0 !important;
            --kmda-inner-bg: #FFFFFF !important;
            --kmda-inner-border: #E2E8F0 !important;

            --kmda-text-primary: #0F172A !important;
            --kmda-text-secondary: #334155 !important;
            --kmda-text-muted: #64748B !important;
            --kmda-divider-grad: linear-gradient(90deg, #E2E8F0 0%, #CBD5E1 50%, #E2E8F0 100%) !important;

            --badge-red-bg: #FEE2E2 !important; --badge-red-text: #DC2626 !important; --badge-red-border: #FECACA !important;
            --badge-amber-bg: #FEF3C7 !important; --badge-amber-text: #D97706 !important; --badge-amber-border: #FDE68A !important;
            --badge-blue-bg: #DBEAFE !important; --badge-blue-text: #2563EB !important; --badge-blue-border: #BFDBFE !important;
            --badge-green-bg: #D1FAE5 !important; --badge-green-text: #059669 !important; --badge-green-border: #A7F3D0 !important;
            --badge-purple-bg: #EDE9FE !important; --badge-purple-text: #7C3AED !important; --badge-purple-border: #DDD6FE !important;
            --badge-slate-bg: #F1F5F9 !important; --badge-slate-text: #475569 !important; --badge-slate-border: #E2E8F0 !important;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        h1, h2, h3, h4 {
            font-family: 'Outfit', sans-serif !important;
            letter-spacing: -0.02em;
        }

        /* ── Adaptive KPI Cards ───────────────────────────────────────────── */
        .kmda-kpi-card {
            background: var(--kmda-card-bg-gradient);
            border: 1px solid var(--kmda-card-border);
            border-radius: 14px;
            padding: 18px 20px;
            box-shadow: var(--kmda-card-shadow);
            transition: all 0.25s ease-in-out;
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        }
        .kmda-kpi-card:hover {
            transform: translateY(-2px);
            box-shadow: var(--kmda-card-shadow-hover);
            border-color: rgba(96, 165, 250, 0.6);
        }
        .kmda-kpi-card::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: #3B82F6;
        }
        .kmda-kpi-card.kpi-red::before { background: #EF4444; }
        .kmda-kpi-card.kpi-amber::before { background: #F59E0B; }
        .kmda-kpi-card.kpi-green::before { background: #10B981; }
        .kmda-kpi-card.kpi-purple::before { background: #8B5CF6; }

        .kmda-kpi-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 6px;
        }
        .kmda-kpi-title {
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--kmda-card-title);
        }
        .kmda-kpi-icon {
            font-size: 1.25rem;
            opacity: 0.90;
        }
        .kmda-kpi-val {
            font-size: 1.95rem;
            font-weight: 800;
            font-family: 'Outfit', sans-serif;
            color: var(--kmda-card-val);
            line-height: 1.15;
            margin-bottom: 4px;
        }
        .kmda-kpi-sub {
            font-size: 0.78rem;
            color: var(--kmda-card-sub);
            display: flex;
            align-items: center;
            gap: 6px;
        }

        /* ── Pill Badges ─────────────────────────────────────────────────── */
        .kmda-badge {
            display: inline-block;
            font-size: 0.70rem;
            font-weight: 700;
            text-transform: uppercase;
            padding: 3px 8px;
            border-radius: 9999px;
            letter-spacing: 0.04em;
            white-space: nowrap;
        }
        .badge-red    { background-color: var(--badge-red-bg); color: var(--badge-red-text); border: 1px solid var(--badge-red-border); }
        .badge-amber  { background-color: var(--badge-amber-bg); color: var(--badge-amber-text); border: 1px solid var(--badge-amber-border); }
        .badge-blue   { background-color: var(--badge-blue-bg); color: var(--badge-blue-text); border: 1px solid var(--badge-blue-border); }
        .badge-green  { background-color: var(--badge-green-bg); color: var(--badge-green-text); border: 1px solid var(--badge-green-border); }
        .badge-purple { background-color: var(--badge-purple-bg); color: var(--badge-purple-text); border: 1px solid var(--badge-purple-border); }
        .badge-slate  { background-color: var(--badge-slate-bg); color: var(--badge-slate-text); border: 1px solid var(--badge-slate-border); }

        /* ── Adaptive Card & Box Classes ─────────────────────────────────── */
        .kmda-box {
            background: var(--kmda-box-bg);
            border: 1px solid var(--kmda-box-border);
            border-radius: 12px;
            padding: 16px 18px;
            color: var(--kmda-text-primary);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
        }

        .kmda-inner-box {
            background: var(--kmda-inner-bg);
            border: 1px solid var(--kmda-inner-border);
            border-radius: 8px;
            padding: 10px 12px;
            color: var(--kmda-text-primary);
        }

        .kmda-attr-card {
            background: var(--kmda-inner-bg);
            border: 1px solid var(--kmda-inner-border);
            border-radius: 10px;
            padding: 12px 14px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .kmda-attr-label {
            font-size: 0.74rem;
            font-weight: 600;
            color: var(--kmda-text-muted);
            text-transform: uppercase;
            margin-bottom: 3px;
        }
        .kmda-attr-val {
            font-size: 1.15rem;
            font-weight: 700;
            font-family: 'Outfit', sans-serif;
            color: var(--kmda-text-primary);
        }

        .kmda-action-card {
            background: var(--kmda-inner-bg);
            border: 1px solid var(--kmda-inner-border);
            border-left-width: 4px;
            border-radius: 8px;
            padding: 14px 16px;
            margin-bottom: 10px;
            color: var(--kmda-text-primary);
        }
        .kmda-action-card.act-high { border-left-color: #DC2626; }
        .kmda-action-card.act-med  { border-left-color: #F59E0B; }
        .kmda-action-card.act-low  { border-left-color: #3B82F6; }

        .kmda-showcase-card {
            background: var(--kmda-inner-bg);
            border: 1px solid var(--kmda-inner-border);
            border-top: 4px solid #DC2626;
            border-radius: 10px;
            padding: 14px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.06);
            height: 100%;
            color: var(--kmda-text-primary);
        }

        .kmda-pill-tag {
            font-size: 0.74rem;
            color: var(--kmda-text-secondary);
            background: var(--kmda-box-bg);
            border: 1px solid var(--kmda-box-border);
            border-radius: 6px;
            padding: 6px 8px;
            margin-top: 8px;
        }

        .kmda-divider {
            height: 1px;
            background: var(--kmda-divider-grad);
            margin: 22px 0 18px 0;
        }
    </style>
    """, unsafe_allow_html=True)


def render_infographic_kpi(title: str, value: str, subtitle: str, icon: str = "📊", color: str = "blue", delta_badge: str = None):
    """Renders a styled infographic metric card compliant with light and dark mode."""
    color_class = f"kpi-{color}" if color in ["red", "amber", "green", "purple"] else ""
    badge_html = f'<span class="kmda-badge badge-{color}">{delta_badge}</span>' if delta_badge else ""

    html = f"""
    <div class="kmda-kpi-card {color_class}">
        <div class="kmda-kpi-header">
            <span class="kmda-kpi-title">{title}</span>
            <span class="kmda-kpi-icon">{icon}</span>
        </div>
        <div class="kmda-kpi-val">{value}</div>
        <div class="kmda-kpi-sub">
            {badge_html}
            <span>{subtitle}</span>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def create_radial_gauge(score: float, title: str = "Segment Risk Score", max_val: float = 100.0, height: int = 240) -> go.Figure:
    """Generates an eye-catching Plotly radial gauge chart that seamlessly complies with both Light and Dark themes."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(score),
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 14, 'family': 'Outfit, sans-serif'}},
        number={'suffix': f" / {int(max_val)}", 'font': {'size': 26, 'family': 'Outfit, sans-serif', 'weight': 'bold'}},
        gauge={
            'axis': {'range': [0, max_val], 'tickwidth': 1, 'tickcolor': "rgba(148, 163, 184, 0.4)"},
            'bar': {'color': "#38BDF8", 'thickness': 0.22},
            'bgcolor': "rgba(128, 128, 128, 0.12)",
            'borderwidth': 1,
            'bordercolor': "rgba(148, 163, 184, 0.25)",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(16, 185, 129, 0.20)'},
                {'range': [40, 70], 'color': 'rgba(245, 158, 11, 0.22)'},
                {'range': [70, max_val], 'color': 'rgba(239, 68, 68, 0.28)'}
            ],
            'threshold': {
                'line': {'color': "#EF4444", 'width': 3},
                'thickness': 0.75,
                'value': 70.0
            }
        }
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=15, r=15, t=35, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif")
    )
    return fig


def render_road_profile_cards(speed_limit: float, lanes: float, has_median: int, is_lit: int, aadt: float = None):
    """Renders a grid of 4 physical roadway cross-section infographic badges compliant with light and dark mode."""
    med_text = "Concrete Barrier" if has_median else "No Barrier / Flush"
    med_color = "badge-green" if has_median else "badge-red"
    lit_text = "High-Mast LED" if is_lit else "Unlit / Dim"
    lit_color = "badge-green" if is_lit else "badge-red"
    speed_color = "badge-red" if speed_limit >= 50 else "badge-blue"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kmda-attr-card">
            <div class="kmda-attr-label">🚗 Speed Limit</div>
            <div class="kmda-attr-val">{int(speed_limit)} km/h</div>
            <span class="kmda-badge {speed_color}">{'High Kinetic' if speed_limit >= 50 else 'Regulated'}</span>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kmda-attr-card">
            <div class="kmda-attr-label">🛣️ Cross-Section</div>
            <div class="kmda-attr-val">{int(lanes)} Lanes</div>
            <span class="kmda-badge badge-blue">{'Multi-Lane' if lanes >= 4 else 'Single-Carriageway'}</span>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kmda-attr-card">
            <div class="kmda-attr-label">🛡️ Median Barrier</div>
            <div class="kmda-attr-val" style="font-size: 0.92rem;">{med_text}</div>
            <span class="kmda-badge {med_color}">{'IRC:119 Compliant' if has_median else 'Head-On Hazard'}</span>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kmda-attr-card">
            <div class="kmda-attr-label">💡 Illumination</div>
            <div class="kmda-attr-val" style="font-size: 0.92rem;">{lit_text}</div>
            <span class="kmda-badge {lit_color}">{'Standard IS 1944' if is_lit else 'Deficient Illumination'}</span>
        </div>
        """, unsafe_allow_html=True)
