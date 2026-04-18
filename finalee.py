"""
MAST IQ — Alloy Intelligence Platform
=======================================
By MAST  ·  designed by mast lab
AI-Powered Steel Heat Treatment Mechanical Property Predictor

Targets  : Tensile · Yield · Hardness · Elongation · Fatigue
Processes: Quench & Temper · Normalizing · Full Annealing · Stress Relief
"""

import os, json as json_lib, base64
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from xgboost import XGBRegressor

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MAST IQ — Alloy Intelligence",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  DESIGN SYSTEM — dual-theme tokens (dark forge + light archive)
# ══════════════════════════════════════════════════════════════════════════════

THEMES = {
    "dark": {
        "bg_base":      "#101114",
        "bg_sink":      "#0A0B0D",
        "bg_elev":      "#181A1E",
        "bg_surface":   "rgba(24,26,30,0.86)",
        "bg_surface_2": "rgba(30,32,37,0.62)",
        "bg_glass":     "rgba(16,17,20,0.78)",
        "border_dim":      "rgba(235,238,244,0.06)",
        "border":          "rgba(235,238,244,0.12)",
        "border_strong":   "rgba(235,238,244,0.22)",
        "fg_primary":   "#EBEEF2",
        "fg_secondary": "#9197A2",
        "fg_muted":     "#5C626E",
        "accent":       "#EBEEF2",
        "accent_dim":   "rgba(235,238,244,0.45)",
        "accent_soft":  "rgba(235,238,244,0.08)",
        "accent_cool":  "#B8BEC8",
        "accent_warn":  "#D4A056",
        "accent_err":   "#C85858",
        "accent_ok":    "#5FB58A",
        "plot_bg":      "rgba(16,17,20,1)",
        "plot_grid":    "rgba(235,238,244,0.06)",
        "plot_text":    "#9197A2",
        "canvas_bg":    "#101114",
        "canvas_panel": "rgba(16,17,20,0.92)",
        "canvas_grid":  "rgba(235,238,244,0.08)",
        "canvas_text":  "rgba(235,238,244,0.62)",
        "hero_a":       "#FFFFFF",
        "hero_b":       "#C4C8D0",
        "hero_c":       "#7A8090",
        "shadow":       "0 4px 24px rgba(0,0,0,0.40)",
        "app_grad":     "#101114",
        "sidebar_grad": "#0E0F12",
        "header_bg":    "rgba(16,17,20,0.88)",
    },
    "light": {
        "bg_base":      "#F6F7F8",
        "bg_sink":      "#EDEEF0",
        "bg_elev":      "#FFFFFF",
        "bg_surface":   "rgba(255,255,255,0.94)",
        "bg_surface_2": "rgba(248,249,251,0.88)",
        "bg_glass":     "rgba(255,255,255,0.82)",
        "border_dim":      "rgba(21,23,28,0.06)",
        "border":          "rgba(21,23,28,0.12)",
        "border_strong":   "rgba(21,23,28,0.22)",
        "fg_primary":   "#15171C",
        "fg_secondary": "#525866",
        "fg_muted":     "#8A91A0",
        "accent":       "#15171C",
        "accent_dim":   "rgba(21,23,28,0.45)",
        "accent_soft":  "rgba(21,23,28,0.06)",
        "accent_cool":  "#525866",
        "accent_warn":  "#A66A18",
        "accent_err":   "#A63838",
        "accent_ok":    "#2E7E54",
        "plot_bg":      "rgba(255,255,255,1)",
        "plot_grid":    "rgba(21,23,28,0.08)",
        "plot_text":    "#525866",
        "canvas_bg":    "#FFFFFF",
        "canvas_panel": "rgba(255,255,255,0.96)",
        "canvas_grid":  "rgba(21,23,28,0.10)",
        "canvas_text":  "rgba(21,23,28,0.72)",
        "hero_a":       "#15171C",
        "hero_b":       "#3A3F4A",
        "hero_c":       "#7A8090",
        "shadow":       "0 2px 14px rgba(21,23,28,0.08)",
        "app_grad":     "#F6F7F8",
        "sidebar_grad": "#FFFFFF",
        "header_bg":    "rgba(255,255,255,0.90)",
    },
}

if "miq_theme" not in st.session_state:
    st.session_state.miq_theme = "dark"

T = THEMES[st.session_state.miq_theme]
IS_DARK = (st.session_state.miq_theme == "dark")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {{
    --bg-base:        {T['bg_base']};
    --bg-sink:        {T['bg_sink']};
    --bg-elev:        {T['bg_elev']};
    --bg-surface:     {T['bg_surface']};
    --bg-surface-2:   {T['bg_surface_2']};
    --bg-glass:       {T['bg_glass']};
    --border-dim:     {T['border_dim']};
    --border:         {T['border']};
    --border-strong:  {T['border_strong']};
    --fg-primary:     {T['fg_primary']};
    --fg-secondary:   {T['fg_secondary']};
    --fg-muted:       {T['fg_muted']};
    --accent:         {T['accent']};
    --accent-dim:     {T['accent_dim']};
    --accent-soft:    {T['accent_soft']};
    --accent-cool:    {T['accent_cool']};
    --accent-warn:    {T['accent_warn']};
    --accent-err:     {T['accent_err']};
    --accent-ok:      {T['accent_ok']};
    --hero-a:         {T['hero_a']};
    --hero-b:         {T['hero_b']};
    --hero-c:         {T['hero_c']};
    --shadow:         {T['shadow']};
}}

/* ══════ Smooth theme transitions ══════ */
html, body, .stApp, [data-testid="stSidebar"], [data-testid="stExpander"],
.miq-card, .miq-pill, .miq-hint, .miq-banner, .miq-meta, .miq-process,
div[data-testid="stAlert"], .stNumberInput input, .stSelectbox > div > div {{
    transition: background-color 0.32s ease, color 0.32s ease,
                border-color 0.32s ease, box-shadow 0.32s ease !important;
}}

html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--fg-primary) !important;
}}

#MainMenu, footer {{ visibility: hidden !important; }}
[data-testid="stDecoration"] {{ display: none !important; }}
[data-testid="stToolbar"] {{ visibility: hidden !important; }}
header[data-testid="stHeader"] {{
    background: {T['header_bg']} !important;
    backdrop-filter: blur(14px) saturate(140%) !important;
    -webkit-backdrop-filter: blur(14px) saturate(140%) !important;
    border-bottom: 1px solid var(--border-dim) !important;
    height: 3.0rem !important;
}}

[data-testid="collapsedControl"] svg {{
    color: var(--accent) !important;
    fill: var(--accent) !important;
}}

/* ══════ App background ══════ */
.stApp {{
    background: {T['app_grad']} !important;
    min-height: 100vh;
}}

[data-testid="stAppViewContainer"] > .main {{ position: relative; z-index: 1; }}

/* ══════ Sidebar ══════ */
[data-testid="stSidebar"] {{
    background: {T['sidebar_grad']} !important;
    border-right: 1px solid var(--border-dim) !important;
    box-shadow: {T['shadow']};
}}
[data-testid="stSidebar"]::before {{
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent-dim), transparent);
}}
[data-testid="stSidebar"] .block-container {{ padding: 1rem 0.95rem !important; }}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] .stMarkdown {{
    color: var(--fg-secondary) !important;
}}
[data-testid="stSidebar"] [data-baseweb="form-control-counter"],
[data-testid="stSidebar"] label {{
    font-size: 0.71rem !important;
    letter-spacing: 0.10em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
}}
[data-testid="stSidebar"] strong {{
    color: var(--fg-primary) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
}}

/* ══════ Inputs ══════ */
.stNumberInput input, .stSelectbox > div > div, .stTextInput input {{
    background: var(--bg-elev) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--fg-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.84rem !important;
}}
.stNumberInput input:focus, .stTextInput input:focus,
.stSelectbox > div > div:focus-within {{
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-soft) !important;
    outline: none !important;
}}
.stNumberInput button {{
    background: transparent !important;
    border: 1px solid var(--border-dim) !important;
    color: var(--fg-secondary) !important;
}}
.stNumberInput button:hover {{
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}}

/* ══════ Sliders ══════ */
[data-testid="stSlider"] > div > div > div > div {{
    background: var(--accent) !important;
    height: 3px !important;
}}
[data-testid="stSlider"] [role="slider"] {{
    background: var(--bg-elev) !important;
    border: 2px solid var(--accent) !important;
    box-shadow: 0 0 0 4px var(--accent-soft) !important;
    width: 16px !important; height: 16px !important;
}}
[data-testid="stSlider"] p {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.76rem !important;
    color: var(--accent) !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
}}

/* ══════ Buttons ══════ */
.stButton > button[kind="primary"] {{
    background: var(--accent) !important;
    border: 1px solid var(--accent) !important;
    color: {('#101114' if IS_DARK else '#FFFFFF')} !important;
    border-radius: 8px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.94rem !important;
    letter-spacing: 0.22em !important;
    text-transform: uppercase !important;
    padding: 0.80rem 1.2rem !important;
    transition: transform 0.18s ease, box-shadow 0.22s ease, filter 0.22s ease, background-color 0.22s ease !important;
    width: 100% !important;
    box-shadow: {('0 4px 14px rgba(0,0,0,0.35)' if IS_DARK else '0 4px 14px rgba(21,23,28,0.18)')} !important;
}}
.stButton > button[kind="primary"]:hover {{
    filter: {('brightness(0.94)' if IS_DARK else 'brightness(1.18)')};
    box-shadow: {('0 6px 18px rgba(0,0,0,0.45)' if IS_DARK else '0 6px 18px rgba(21,23,28,0.28)')} !important;
    transform: translateY(-1px);
}}
.stButton > button[kind="primary"]:active {{ transform: translateY(0); filter: brightness(0.96); }}

.stButton > button[kind="secondary"] {{
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--fg-primary) !important;
    border-radius: 8px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
}}
.stButton > button[kind="secondary"]:hover {{
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: var(--accent-soft) !important;
}}

/* ══════ Compact theme toggle — targets button by Streamlit key ══════ */
.st-key-miq_theme_toggle {{
    display: flex !important;
    justify-content: flex-end !important;
    margin-top: 1.6rem !important;
}}
.st-key-miq_theme_toggle .stButton > button,
.st-key-miq_theme_toggle button {{
    width: auto !important;
    min-width: 0 !important;
    padding: 0.42rem 0.95rem !important;
    border-radius: 999px !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--fg-secondary) !important;
    box-shadow: var(--shadow) !important;
    transition: all 0.25s ease !important;
}}
.st-key-miq_theme_toggle .stButton > button:hover,
.st-key-miq_theme_toggle button:hover {{
    color: var(--accent) !important;
    border-color: var(--accent-dim) !important;
    background: var(--accent-soft) !important;
    transform: translateY(-1px) !important;
}}

/* ══════ Tabs ══════ */
.stTabs [data-baseweb="tab-list"] {{
    background: transparent !important;
    border: none !important;
    border-bottom: 1px solid var(--border-dim) !important;
    border-radius: 0 !important;
    padding: 0 !important;
    gap: 0 !important;
    overflow-x: auto !important;
    flex-wrap: nowrap !important;
    -webkit-overflow-scrolling: touch !important;
    scrollbar-width: none !important;
}}
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {{ display: none !important; }}
.stTabs [data-baseweb="tab"] {{
    border-radius: 0 !important;
    background: transparent !important;
    color: var(--fg-muted) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    padding: 0.72rem 1.20rem !important;
    white-space: nowrap !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.18s !important;
    flex-shrink: 0 !important;
}}
.stTabs [data-baseweb="tab"]:hover {{ color: var(--fg-primary) !important; }}
.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    box-shadow: none !important;
}}

/* ══════ Expander ══════ */
[data-testid="stExpander"] {{
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-dim) !important;
    border-radius: 8px !important;
}}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p {{
    color: var(--fg-secondary) !important;
    font-size: 0.82rem !important;
}}

/* ══════ DataFrame ══════ */
.stDataFrame {{
    border-radius: 8px !important;
    overflow: hidden !important;
    border: 1px solid var(--border-dim) !important;
}}

hr {{ border: none !important; border-top: 1px solid var(--border-dim) !important; margin: 14px 0 !important; }}

iframe {{ max-width: 100% !important; border-radius: 8px !important; }}

.block-container {{
    padding: 1.4rem 2.2rem 1rem !important;
    max-width: 1480px !important;
    position: relative;
    z-index: 1;
}}

/* ══════ Alerts ══════ */
div[data-testid="stAlert"] {{
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-dim) !important;
    border-left: 3px solid var(--accent) !important;
    border-radius: 6px !important;
    color: var(--fg-primary) !important;
}}
div[data-testid="stAlert"] p {{ color: var(--fg-secondary) !important; }}

/* ══════ Custom layout primitives ══════ */
.miq-card {{
    background: var(--bg-surface);
    border: 1px solid var(--border-dim);
    border-radius: 12px;
    padding: 16px 18px;
    box-shadow: var(--shadow);
}}

.miq-cards {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 12px;
    margin: 0 0 16px;
}}

.miq-pill {{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px;
    border-radius: 999px;
    background: var(--accent-soft);
    border: 1px solid var(--accent-dim);
    color: var(--accent);
    font-size: 0.71rem;
    font-weight: 600;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    font-family: 'Rajdhani', sans-serif;
}}

.miq-hint {{
    display: none;
    align-items: center;
    gap: 10px;
    background: var(--bg-surface);
    border: 1px solid var(--border-dim);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 10px 14px;
    margin: 46px 0 14px;
    font-size: 0.78rem;
    color: var(--fg-secondary);
}}
.miq-hint span:first-child {{ color: var(--accent); font-size: 1.1rem; }}

/* ══════════════════════════════════════════════════════════
   MOBILE-FIRST RESPONSIVE BREAKPOINTS
   ══════════════════════════════════════════════════════════ */
@media (max-width: 1280px) {{
    .block-container {{ padding: 1.1rem 1.4rem 0.9rem !important; max-width: 100% !important; }}
}}

@media (max-width: 1024px) {{
    .block-container {{ padding: 0.95rem 1.05rem 0.75rem !important; }}
    .miq-hero-title {{ font-size: 2.6rem !important; }}
}}

@media (max-width: 768px) {{
    .block-container {{
        padding: 0.8rem 0.85rem 0.65rem !important;
        max-width: 100vw !important;
    }}
    [data-testid="stHorizontalBlock"],
    [data-testid="stColumns"] {{
        flex-wrap: wrap !important;
        gap: 0.4rem !important;
    }}
    [data-testid="column"],
    [data-testid="stColumn"] {{
        min-width: 100% !important;
        width: 100% !important;
        flex: 1 1 100% !important;
    }}
    [data-testid="stSidebar"] {{
        min-width: 280px !important;
        max-width: 88vw !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-size: 0.70rem !important;
        padding: 0.55rem 0.85rem !important;
        letter-spacing: 0.08em !important;
    }}
    .miq-hero-title {{ font-size: 2.1rem !important; letter-spacing: 0.05em !important; }}
    .miq-hero-sub   {{ font-size: 0.62rem !important; letter-spacing: 0.18em !important; }}
    .miq-cards {{
        grid-template-columns: repeat(2, 1fr) !important;
        gap: 10px !important;
    }}
    .miq-hint {{ display: flex !important; }}
    [data-testid="stAppViewContainer"] > .main {{ padding-top: 0.4rem !important; }}
}}

@media (max-width: 480px) {{
    .block-container {{ padding: 0.55rem 0.6rem 0.5rem !important; }}
    .miq-hero-title {{ font-size: 1.7rem !important; }}
    .miq-hero-sub   {{ font-size: 0.56rem !important; letter-spacing: 0.14em !important; }}
    .miq-cards {{ grid-template-columns: 1fr !important; }}
    .stTabs [data-baseweb="tab"] {{
        font-size: 0.64rem !important;
        padding: 0.5rem 0.6rem !important;
    }}
    .stNumberInput input {{ font-size: 0.8rem !important; }}
}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HTML HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def html_metric_card(title, value, unit, color, icon, subtitle=""):
    sub = (f'<div style="font-size:0.66rem;color:var(--fg-muted);margin-top:3px;">'
           f'{subtitle}</div>') if subtitle else ""
    return (
        f'<div class="miq-card" style="border-top:3px solid {color};'
        f'padding:16px 12px 14px;text-align:center;">'
        f'<div style="font-size:1.35rem;margin-bottom:5px;line-height:1;">{icon}</div>'
        f'<div style="font-size:2.05rem;font-weight:700;color:{color};'
        f'font-family:\'Rajdhani\',sans-serif;line-height:1.0;">{value}</div>'
        f'<div style="font-size:0.62rem;color:var(--fg-secondary);text-transform:uppercase;'
        f'letter-spacing:0.16em;margin-top:6px;font-weight:600;">{title}</div>'
        f'<div style="font-size:0.68rem;color:var(--fg-muted);margin-top:2px;">{unit}</div>'
        f'{sub}</div>'
    )


def html_section_header(title, subtitle="", icon=""):
    sub = (f'<p style="color:var(--fg-secondary);font-size:0.82rem;margin:0 0 0 2.0rem;">'
           f'{subtitle}</p>') if subtitle else ""
    return (
        f'<div style="margin:0 0 16px 0;">'
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:3px;">'
        f'<span style="font-size:1.25rem;">{icon}</span>'
        f'<span style="font-family:\'Rajdhani\',sans-serif;font-size:1.32rem;'
        f'font-weight:700;color:var(--fg-primary);letter-spacing:0.04em;">{title}</span>'
        f'</div>{sub}</div>'
    )


def html_badge(text, color):
    return (f'<span style="display:inline-block;padding:3px 11px;border-radius:999px;'
            f'background:{color}22;border:1px solid {color}55;color:{color};'
            f'font-size:0.71rem;font-weight:600;letter-spacing:0.08em;'
            f'text-transform:uppercase;">{text}</span>')


def html_process_card(proc, color, icon, desc):
    return (
        f'<div class="miq-card" style="border-left:3px solid {color};margin-bottom:6px;'
        f'padding:14px 16px;">'
        f'<div style="display:flex;align-items:center;gap:9px;margin-bottom:6px;">'
        f'<span style="font-size:1.10rem;">{icon}</span>'
        f'<span style="font-family:\'Rajdhani\',sans-serif;font-size:1.05rem;'
        f'font-weight:700;color:{color};letter-spacing:0.04em;">{proc}</span></div>'
        f'<div style="font-size:0.76rem;color:var(--fg-secondary);line-height:1.5;">{desc}</div></div>'
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PHYSICAL METALLURGY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def calc_a3(C, Mn=0.0, Si=0.0, Ni=0.0, Cr=0.0, Mo=0.0):
    return 912.0 - 203.0 * np.sqrt(max(C, 1e-6)) - 30.0*Mn + 44.7*Si - 15.2*Ni + 31.5*Mo

def calc_carbon_equiv(C, Mn, Si, Ni, Cr, Mo, Cu):
    return C + Mn/6 + Si/24 + Ni/40 + Cr/5 + Mo/4 + Cu/15

def calc_hollomon_jaffe(T_C, t_min):
    if T_C <= 0 or t_min <= 0:
        return 0.0
    return (T_C + 273.15) * (18.0 + np.log10(max(t_min / 60.0, 0.001)))

def get_phase(c, T):
    A1=723; A3_FE=912; EUTECTIC_T=1147; C_EUT=0.76; C_AUS_MAX=2.11
    def a3(cc): return A3_FE - (A3_FE-A1)/C_EUT * min(cc, C_EUT)
    def acm(cc): return A1 + (EUTECTIC_T-A1)/(C_AUS_MAX-C_EUT) * (cc-C_EUT)
    if T >= 1400: return "Liquid"
    if T >= A1:
        if c <= C_EUT and T < a3(c):  return "Austenite (γ) + Ferrite (α)"
        if c > C_EUT and T < acm(c):  return "Austenite (γ) + Cementite (Fe₃C)"
        return "Austenite (γ)"
    if c <= 0.022: return "Ferrite (α)"
    if c <= C_EUT:  return "Ferrite (α) + Pearlite"
    return "Pearlite + Cementite (Fe₃C)"

PHASE_COLORS = {
    "Austenite (γ)":                    "#ffd700",
    "Austenite (γ) + Ferrite (α)":      "#ff8c30",
    "Austenite (γ) + Cementite (Fe₃C)": "#ff5c6a",
    "Ferrite (α)":                      "#50c878",
    "Ferrite (α) + Pearlite":           "#56b4d3",
    "Pearlite + Cementite (Fe₃C)":      "#b07ad6",
    "Liquid":                           "#f0e68c",
}


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
TARGETS = ["Tensile_MPa", "Yield_MPa", "Hardness_HB", "Elongation_pct", "Fatigue_MPa"]
TARGET_LABELS = {
    "Tensile_MPa":    ("Tensile Strength", "MPa", "💪", "#4c72b0"),
    "Yield_MPa":      ("Yield Strength",   "MPa", "🔩", "#f0a030"),
    "Hardness_HB":    ("Hardness",         "HB",  "💎", "#50c878"),
    "Elongation_pct": ("Elongation",       "%",   "📐", "#e05060"),
    "Fatigue_MPa":    ("Fatigue Strength", "MPa", "🔄", "#9370db"),
}
PROP_RANGES = {
    "Tensile_MPa":    (300,  2200),
    "Yield_MPa":      (150,  1900),
    "Hardness_HB":    (80,   680),
    "Elongation_pct": (3,    48),
    "Fatigue_MPa":    (100,  1000),
}


@st.cache_resource(show_spinner="Loading MAST IQ models…")
def load_models():
    models_dir = "models"
    metadata   = None
    for meta_path in ["model_metrics.json",
                       os.path.join(models_dir, "model_metrics.json")]:
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                metadata = json_lib.load(f)
            break
    per_target = {}
    for tgt in TARGETS:
        for p in [f"xgb_{tgt.lower()}.json",
                  os.path.join(models_dir, f"xgb_{tgt.lower()}.json")]:
            if os.path.exists(p):
                m = XGBRegressor(); m.load_model(p)
                per_target[tgt] = m
                break
    if len(per_target) == len(TARGETS):
        return {"mode": "per_target", "models": per_target, "metadata": metadata}
    if os.path.exists("xgb_model.json"):
        m = XGBRegressor(); m.load_model("xgb_model.json")
        return {"mode": "legacy", "models": m, "metadata": None}
    st.error("No model found. Run the training notebook first.")
    st.stop()


model_pack   = load_models()
MODEL_MODE   = model_pack["mode"]
META         = model_pack["metadata"] or {}
FEATURE_COLS = META.get("features", [])
TRAINING_RANGES = {
    k: (v["min"], v["max"])
    for k, v in META.get("training_ranges", {}).items()
}

PROCESS_DUMMIES = {
    "Quench_Temper": {"Process_Quench_Temper":1,"Process_Normalizing":0,"Process_Annealing":0,"Process_Stress_Relief":0},
    "Normalizing":   {"Process_Quench_Temper":0,"Process_Normalizing":1,"Process_Annealing":0,"Process_Stress_Relief":0},
    "Annealing":     {"Process_Quench_Temper":0,"Process_Normalizing":0,"Process_Annealing":1,"Process_Stress_Relief":0},
    "Stress_Relief": {"Process_Quench_Temper":0,"Process_Normalizing":0,"Process_Annealing":0,"Process_Stress_Relief":1},
}
MEDIUM_DUMMIES = {
    "Water":    {"Cooling_Medium_Water":1,"Cooling_Medium_Oil":0,"Cooling_Medium_Polymer":0,"Cooling_Medium_Air":0,"Cooling_Medium_Furnace":0,"Cooling_Medium_Salt Bath":0},
    "Oil":      {"Cooling_Medium_Water":0,"Cooling_Medium_Oil":1,"Cooling_Medium_Polymer":0,"Cooling_Medium_Air":0,"Cooling_Medium_Furnace":0,"Cooling_Medium_Salt Bath":0},
    "Polymer":  {"Cooling_Medium_Water":0,"Cooling_Medium_Oil":0,"Cooling_Medium_Polymer":1,"Cooling_Medium_Air":0,"Cooling_Medium_Furnace":0,"Cooling_Medium_Salt Bath":0},
    "Air":      {"Cooling_Medium_Water":0,"Cooling_Medium_Oil":0,"Cooling_Medium_Polymer":0,"Cooling_Medium_Air":1,"Cooling_Medium_Furnace":0,"Cooling_Medium_Salt Bath":0},
    "Furnace":  {"Cooling_Medium_Water":0,"Cooling_Medium_Oil":0,"Cooling_Medium_Polymer":0,"Cooling_Medium_Air":0,"Cooling_Medium_Furnace":1,"Cooling_Medium_Salt Bath":0},
    "Salt Bath":{"Cooling_Medium_Water":0,"Cooling_Medium_Oil":0,"Cooling_Medium_Polymer":0,"Cooling_Medium_Air":0,"Cooling_Medium_Furnace":0,"Cooling_Medium_Salt Bath":1},
}
COOLING_RATE_EST = {
    "Water": 80.0, "Oil": 25.0, "Polymer": 35.0,
    "Salt Bath": 15.0, "Air": 2.0, "Furnace": 0.1,
}


def build_feature_vector(process_key, C, Si, Mn, P, S, Ni, Cr, Cu, Mo,
                          ht_temp, soak_time, cool_medium,
                          t_temp=0.0, t_time=0.0):
    CE  = calc_carbon_equiv(C, Mn, Si, Ni, Cr, Mo, Cu)
    A3  = calc_a3(C, Mn, Si, Ni, Cr, Mo)
    HJ  = calc_hollomon_jaffe(t_temp, t_time)
    Fe  = 100.0 - (C + Si + Mn + P + S + Ni + Cr + Cu + Mo)
    total_alloy = C + Si + Mn + Ni + Cr + Mo + Cu
    cool_rate = COOLING_RATE_EST.get(cool_medium, 2.0)
    feat = {
        "C": C, "Si": Si, "Mn": Mn, "P": P, "S": S,
        "Ni": Ni, "Cr": Cr, "Cu": Cu, "Mo": Mo, "Fe": Fe,
        "Carbon_Equiv": CE, "A3_Temp_C": A3, "Delta_HT_A3": ht_temp - A3,
        "Hollomon_Jaffe": HJ, "C_x_Cr": C * Cr,
        "Cooling_Rate_Est": cool_rate, "Total_Alloy": total_alloy,
        "HT_Temp_C": ht_temp, "Soaking_Time_min": soak_time,
        "Tempering_Temp_C": t_temp, "Tempering_Time_min": t_time,
        **PROCESS_DUMMIES.get(process_key, PROCESS_DUMMIES["Quench_Temper"]),
        **MEDIUM_DUMMIES.get(cool_medium, MEDIUM_DUMMIES["Air"]),
    }
    return feat


def predict(feat_dict):
    cols = FEATURE_COLS if FEATURE_COLS else list(feat_dict.keys())
    for c in cols:
        if c not in feat_dict:
            feat_dict[c] = 0
    df_in = pd.DataFrame([feat_dict])[cols].astype(float)
    if MODEL_MODE == "per_target":
        return {t: max(0.0, float(model_pack["models"][t].predict(df_in)[0]))
                for t in TARGETS}
    raw = model_pack["models"].predict(df_in)[0]
    return {t: max(0.0, float(v)) for t, v in zip(TARGETS[:3], raw)}


def check_ood(feat_dict):
    return [f"**{k}** = {feat_dict[k]:.3f}  (training range: [{lo:.2f}, {hi:.2f}])"
            for k, (lo, hi) in TRAINING_RANGES.items()
            if k in feat_dict and (feat_dict[k] < lo or feat_dict[k] > hi)]


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTLY CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════
_BASE = dict(
    plot_bgcolor =T['plot_bg'],
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color=T['plot_text'], family="Inter, sans-serif", size=11),
)
_PLOT_GRID = T['plot_grid']
_PLOT_TEXT = T['plot_text']


def build_gauges(preds):
    specs = [[{"type": "indicator"}] * 5]
    fig   = make_subplots(rows=1, cols=5, specs=specs)
    for i, tgt in enumerate(TARGETS, 1):
        label, unit, icon, col = TARGET_LABELS[tgt]
        lo, hi = PROP_RANGES[tgt]
        val    = preds[tgt]
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=val,
            number=dict(font=dict(size=22, color=col, family="Rajdhani, sans-serif"),
                        suffix=f" {unit}"),
            title=dict(text=f"<b>{label}</b>", font=dict(size=9, color="#b0cef0")),
            gauge=dict(
                axis=dict(range=[lo, hi], tickwidth=1,
                          tickcolor="rgba(180,210,255,0.28)",
                          tickfont=dict(size=7, color="rgba(180,210,255,0.35)")),
                bar=dict(color=col, thickness=0.26),
                bgcolor="rgba(4,7,18,0)", borderwidth=0,
                steps=[
                    dict(range=[lo, lo+(hi-lo)*0.33], color="rgba(255,255,255,0.02)"),
                    dict(range=[lo+(hi-lo)*0.33, lo+(hi-lo)*0.66], color="rgba(255,255,255,0.04)"),
                    dict(range=[lo+(hi-lo)*0.66, hi], color="rgba(255,255,255,0.07)"),
                ],
                threshold=dict(line=dict(color="rgba(255,255,255,0.8)", width=2),
                               thickness=0.75, value=val),
            ),
        ), row=1, col=i)
    fig.update_layout(**_BASE, height=245, margin=dict(l=18, r=18, t=38, b=8))
    return fig


def build_radar(preds):
    cats, vals = [], []
    for tgt in TARGETS:
        lo, hi = PROP_RANGES[tgt]
        cats.append(TARGET_LABELS[tgt][0])
        vals.append(float(np.clip((preds[tgt]-lo)/(hi-lo)*100, 0, 100)))
    cats_c = cats + [cats[0]]
    vals_c  = vals + [vals[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_c, theta=cats_c, fill="toself",
        fillcolor="rgba(0,100,200,0.18)",
        line=dict(color="#00aaff", width=2.5),
        marker=dict(size=7, color="#60d0ff"),
        name="Predicted",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[50]*6, theta=cats_c, fill="toself",
        fillcolor="rgba(255,255,255,0.03)",
        line=dict(color="rgba(255,255,255,0.18)", width=1, dash="dot"),
        name="Dataset avg.",
    ))
    fig.update_layout(
        **_BASE,
        polar=dict(
            bgcolor="rgba(4,7,18,0.5)",
            radialaxis=dict(range=[0,100], ticksuffix="%",
                            gridcolor="rgba(180,210,255,0.09)",
                            linecolor="rgba(180,210,255,0.09)",
                            tickfont=dict(size=7, color="rgba(180,210,255,0.38)")),
            angularaxis=dict(gridcolor="rgba(180,210,255,0.09)",
                             linecolor="rgba(180,210,255,0.14)",
                             tickfont=dict(size=10, color="#b0cef0")),
        ),
        showlegend=True,
        legend=dict(bgcolor="rgba(4,7,18,0.75)", bordercolor="rgba(0,150,255,0.28)",
                    borderwidth=1, font=dict(size=10)),
        height=320, margin=dict(l=38, r=38, t=28, b=18),
    )
    return fig


def build_phase_diagram(C_val, ht_temp, t_temp, process_key, cool_medium,
                          Mn=0.85, Cr=1.05, Mo=0.20):
    """
    Immersive Fe-C phase diagram:
    - Thermal heatmap background (liquid metal glows, solid phases subdued)
    - Translucent phase-region fills with distinct colors
    - Sharp phase-boundary lines (A3, Acm, A1, Liquidus, Solidus, Ms, Bs)
    - Process temperature lines, carbon line, heating/cooling arrows
    - Operating point with multi-ring glow effect
    - Special points: Eutectoid (S), Peritectic (P)
    - Horizontal legend placed below the chart to avoid overlap
    """
    A1=723; A3_FE=912; PERIT=1495; MELT=1538; EUTE=1147
    C_EUT=0.76; C_AUS=2.11; MAX_C=2.3; C_PER=0.17

    def _a3(c):  return A3_FE - (A3_FE-A1)/C_EUT * min(c, C_EUT)
    def _acm(c): return A1 + (EUTE-A1)/(C_AUS-C_EUT) * (c-C_EUT)

    Ms_val = max(80.0,  539 - 423*C_val - 30.4*Mn - 12.1*Cr - 7.5*Mo)
    Bs_val = max(300.0, 720 - 270*C_val - 90*Mn  - 70*Cr)

    # ── Build thermal heatmap background (vectorized) ──
    nC, nT = 220, 300
    C_arr = np.linspace(0, MAX_C, nC)
    T_arr = np.linspace(0, 1650, nT)
    CC, TT = np.meshgrid(C_arr, T_arr)

    A3_V  = np.where(CC <= C_EUT, A3_FE - (A3_FE-A1)/C_EUT * CC, A1)
    ACM_V = np.where(CC > C_EUT,  A1 + (EUTE-A1)/(C_AUS-C_EUT) * (CC - C_EUT), EUTE)

    Z = np.full_like(CC, 0.08, dtype=float)

    m = (TT < A1) & (CC <= 0.022)
    Z[m] = 0.08
    m = (TT < A1) & (CC > 0.022) & (CC <= C_EUT)
    Z[m] = 0.14
    m = (TT < A1) & (CC > C_EUT)
    Z[m] = 0.11

    m_ga = (CC <= C_EUT) & (TT >= A1) & (TT < A3_V)
    Z[m_ga] = 0.30
    m_gc = (CC > C_EUT) & (CC <= C_AUS) & (TT >= A1) & (TT < ACM_V)
    Z[m_gc] = 0.28

    m_au = ((CC <= C_EUT) & (TT >= A3_V)) | \
           ((CC > C_EUT)  & (CC <= C_AUS) & (TT >= ACM_V))
    m_au &= (TT < 1280)
    Z[m_au] = np.clip(0.44 + (TT[m_au] - 720) / 1800, 0.44, 0.62)

    m_hot = (TT >= 1280) & (TT < MELT) & (CC <= C_AUS)
    Z[m_hot] = np.clip(0.64 + (TT[m_hot] - 1280) / 1000, 0.64, 0.83)

    m_dl = (CC <= C_PER) & (TT >= PERIT) & (TT < MELT + 80)
    Z[m_dl] = np.clip(0.80 + (TT[m_dl] - PERIT) / 900, 0.80, 0.92)

    m_liq = (TT >= MELT) | \
            ((TT >= PERIT) & (CC <= C_PER) & (TT >= MELT - 40)) | \
            ((TT >= EUTE)  & (CC >= C_AUS))
    Z[m_liq] = np.clip(0.90 + (TT[m_liq] - EUTE) / 1600, 0.90, 1.0)

    Z = np.clip(Z, 0.0, 1.0)

    THERMAL_CS = [
        [0.00, "rgb(2,3,12)"],
        [0.08, "rgb(5,7,22)"],
        [0.14, "rgb(10,12,30)"],
        [0.22, "rgb(18,14,18)"],
        [0.32, "rgb(36,18,10)"],
        [0.44, "rgb(66,22,2)"],
        [0.55, "rgb(110,40,0)"],
        [0.65, "rgb(165,65,2)"],
        [0.75, "rgb(215,100,8)"],
        [0.85, "rgb(248,150,25)"],
        [0.92, "rgb(255,200,70)"],
        [1.00, "rgb(255,245,185)"],
    ]

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        x=C_arr, y=T_arr, z=Z,
        colorscale=THERMAL_CS, showscale=False,
        zmin=0.0, zmax=1.0,
        hoverinfo="skip",
    ))

    # ── Phase region color fills (translucent overlay) ──
    regions = [
        ([0, 0.022, 0.022, 0],            [0,0,A1,A1],
         "rgba(80,200,120,0.28)",  "Ferrite (α)"),
        ([0.022,C_EUT,C_EUT,0.022],       [0,0,A1,A1],
         "rgba(86,180,211,0.22)",  "Ferrite + Pearlite"),
        ([C_EUT,MAX_C,MAX_C,C_EUT],       [0,0,A1,A1],
         "rgba(176,122,214,0.22)", "Pearlite + Fe₃C"),
        ([0,C_EUT,0],                     [A3_FE,A1,A1],
         "rgba(255,140,48,0.20)",  "Austenite + Ferrite"),
        ([C_EUT,MAX_C,MAX_C,C_AUS],       [A1,A1,EUTE,EUTE],
         "rgba(255,92,106,0.20)",  "Austenite + Fe₃C"),
        ([0,C_PER,C_AUS,C_EUT,0],         [MELT,PERIT,EUTE,A1,A3_FE],
         "rgba(255,215,0,0.14)",   "Austenite (γ)"),
        ([0,C_PER,C_PER,0],               [MELT,PERIT,1640,1640],
         "rgba(170,225,255,0.22)", "δ-Ferrite + Liquid"),
        ([0,MAX_C,MAX_C,C_AUS,C_PER,0],   [MELT,EUTE,1650,1650,PERIT,MELT],
         "rgba(255,245,180,0.10)", "Liquid"),
    ]
    for xs, ys, fill, lbl in regions:
        fig.add_trace(go.Scatter(
            x=xs, y=ys, fill="toself", fillcolor=fill,
            line=dict(width=0),
            mode="lines", showlegend=False,
            hovertemplate=f"<b>{lbl}</b><br>C=%{{x:.3f}} wt%  T=%{{y:.0f}} °C<extra></extra>",
        ))

    # ── Phase boundary lines ──
    liq_c = np.linspace(0, MAX_C, 400)
    liq_T = [MELT-(MELT-PERIT)/C_PER*c if c<=C_PER else
             PERIT+(EUTE-PERIT)/(C_AUS-C_PER)*(c-C_PER) if c<=C_AUS else
             EUTE+(1640-EUTE)/(MAX_C-C_AUS)*(c-C_AUS)
             for c in liq_c]
    fig.add_trace(go.Scatter(
        x=liq_c, y=liq_T,
        line=dict(color="rgba(255,245,170,0.90)", width=2.2),
        mode="lines", name="Liquidus",
        hovertemplate="Liquidus: %{y:.0f} °C<extra></extra>"))

    a3_c = np.linspace(0, C_EUT, 200)
    fig.add_trace(go.Scatter(
        x=a3_c, y=[_a3(c) for c in a3_c],
        line=dict(color="#f5b840", width=2.2),
        mode="lines", name="A3 line",
        hovertemplate="A3 = %{y:.0f} °C<extra></extra>"))

    acm_c = np.linspace(C_EUT, C_AUS, 200)
    fig.add_trace(go.Scatter(
        x=acm_c, y=[_acm(c) for c in acm_c],
        line=dict(color="#e85060", width=2.2),
        mode="lines", name="Acm line",
        hovertemplate="Acm = %{y:.0f} °C<extra></extra>"))

    fig.add_trace(go.Scatter(
        x=[0, MAX_C], y=[A1, A1],
        line=dict(color="#88aaff", width=2.0, dash="dot"),
        mode="lines", name="A1 = 723 °C"))

    # Solidus (δ region lower boundary)
    fig.add_trace(go.Scatter(
        x=[0, C_PER], y=[MELT, PERIT],
        line=dict(color="rgba(160,220,255,0.60)", width=1.7, dash="dot"),
        mode="lines", name="δ Solidus"))

    # Ms
    fig.add_trace(go.Scatter(
        x=[0, C_EUT+0.2], y=[Ms_val, Ms_val],
        line=dict(color="rgba(170,150,255,0.78)", width=1.6, dash="dashdot"),
        mode="lines",
        name=f"Ms = {Ms_val:.0f} °C",
        hovertemplate="Martensite start: %{y:.0f} °C<extra></extra>"))

    # Bs
    if Bs_val > Ms_val + 30:
        fig.add_trace(go.Scatter(
            x=[0, C_EUT+0.2], y=[Bs_val, Bs_val],
            line=dict(color="rgba(200,150,255,0.58)", width=1.3, dash="dashdot"),
            mode="lines",
            name=f"Bs = {Bs_val:.0f} °C",
            hovertemplate="Bainite start: %{y:.0f} °C<extra></extra>"))

    # ── Special invariant points ──
    fig.add_trace(go.Scatter(
        x=[C_EUT], y=[A1], mode="markers",
        marker=dict(size=12, color="#ffd700", symbol="diamond",
                    line=dict(color="white", width=2)),
        name="S — Eutectoid",
        hovertemplate="<b>Eutectoid (S)</b><br>0.76 wt%  ·  723 °C<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=[C_PER], y=[PERIT], mode="markers",
        marker=dict(size=12, color="#60c4ff", symbol="triangle-up",
                    line=dict(color="white", width=2)),
        name="P — Peritectic",
        hovertemplate="<b>Peritectic (P)</b><br>0.17 wt%  ·  1495 °C<extra></extra>"))

    # ── Process + carbon lines (no legend entries — labels on-plot) ──
    fig.add_trace(go.Scatter(
        x=[0, MAX_C], y=[ht_temp, ht_temp],
        line=dict(color="#00d8ff", width=2.0, dash="dashdot"),
        mode="lines+text",
        text=["", f"Austenitize {ht_temp:.0f}°C"],
        textposition="middle right",
        textfont=dict(size=9, color="#00d8ff"),
        name="Austenitize", showlegend=False))
    if t_temp > 0:
        fig.add_trace(go.Scatter(
            x=[0, MAX_C], y=[t_temp, t_temp],
            line=dict(color="#ffb030", width=1.6, dash="dashdot"),
            mode="lines+text",
            text=["", f"Temper {t_temp:.0f}°C"],
            textposition="middle right",
            textfont=dict(size=9, color="#ffb030"),
            name="Temper", showlegend=False))
    fig.add_trace(go.Scatter(
        x=[C_val, C_val], y=[0, 1640],
        line=dict(color="rgba(80,230,190,0.70)", width=1.6, dash="dot"),
        mode="lines+text",
        text=[f"C={C_val:.2f}%", ""],
        textposition="top right",
        textfont=dict(size=9, color="#50e3c2"),
        name=f"C = {C_val:.2f}%", showlegend=False))

    # ── Heating + cooling arrows ──
    ax_off = min(C_val + 0.08, MAX_C - 0.14)
    COOL_COLS = {"Water":"#60c4ff","Oil":"#f0a030","Polymer":"#c860ff",
                 "Air":"#50e3c2","Furnace":"#ff6b35","Salt Bath":"#ffd700"}
    cool_col = COOL_COLS.get(cool_medium, "#aaaaaa")
    end_T = Ms_val if cool_medium in ["Water","Oil","Polymer","Salt Bath"] else 200

    fig.add_annotation(
        x=ax_off, y=ht_temp, ax=ax_off, ay=80,
        xref="x", yref="y", axref="x", ayref="y",
        arrowhead=4, arrowcolor="#ff8844", arrowwidth=3.0, arrowside="end",
        text="", showarrow=True)
    fig.add_annotation(
        x=ax_off+0.10, y=end_T, ax=ax_off+0.10, ay=ht_temp,
        xref="x", yref="y", axref="x", ayref="y",
        arrowhead=4, arrowcolor=cool_col, arrowwidth=3.0, arrowside="end",
        text="", showarrow=True)

    # ── Operating point with multi-ring glow ──
    op_phase = get_phase(C_val, ht_temp)
    op_color = PHASE_COLORS.get(op_phase, "#ffffff")
    for sz, opa in [(34, 0.07), (24, 0.13), (16, 0.22)]:
        fig.add_trace(go.Scatter(
            x=[C_val], y=[ht_temp], mode="markers",
            marker=dict(size=sz, color=op_color, opacity=opa),
            showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[C_val], y=[ht_temp], mode="markers",
        marker=dict(size=14, color=op_color, symbol="circle",
                    line=dict(color="white", width=2.6)),
        name="Operating point",
        hovertemplate=(f"<b>Austenitizing condition</b><br>"
                       f"C = {C_val:.3f} wt%<br>T = {ht_temp:.0f} °C<br>"
                       f"Phase: <b>{op_phase}</b><extra></extra>")))

    # ── Phase region labels ──
    for lx, ly, ltxt in [
        (0.011, 200, "α"),
        (0.38,  200, "α + P"),
        (1.55,  200, "P + Fe₃C"),
        (0.22,  790, "γ + α"),
        (1.85,  900, "γ + Fe₃C"),
        (0.55, 1060, "γ"),
        (1.10, 1380, "Liquid"),
        (0.06, 1520, "δ + L"),
    ]:
        fig.add_annotation(
            x=lx, y=ly, text=f"<b>{ltxt}</b>",
            showarrow=False,
            font=dict(size=11, color="rgba(230,240,255,0.82)"),
            bgcolor="rgba(0,0,0,0)", xref="x", yref="y")

    fig.update_layout(
        **_BASE,
        title=dict(
            text=(f"<b>Fe-C Phase Diagram</b>  ·  "
                  f"Current phase: <span style='color:{op_color}'>{op_phase}</span>"),
            font=dict(size=13), x=0.5, xanchor="center"),
        xaxis=dict(
            title="Carbon Content (wt%)",
            range=[0, MAX_C],
            gridcolor="rgba(180,210,255,0.05)",
            zeroline=False,
            tickfont=dict(size=10),
            tickformat=".2f"),
        yaxis=dict(
            title="Temperature (°C)",
            range=[0, 1650],
            gridcolor="rgba(180,210,255,0.05)",
            zeroline=False,
            tickfont=dict(size=10)),
        legend=dict(
            orientation="h",
            x=0.5, y=-0.16,
            xanchor="center", yanchor="top",
            bgcolor="rgba(4,10,24,0.88)",
            bordercolor="rgba(61,184,255,0.22)",
            borderwidth=1,
            font=dict(size=9, color="#c0d8f0"),
            itemwidth=30,
        ),
        height=680,
        margin=dict(l=62, r=62, t=58, b=150),
    )
    return fig


def build_immersive_phase_html(C_val, ht_temp, t_temp, process_key, cool_medium,
                                Mn=0.85, Cr=1.05, Mo=0.20):
    """Canvas-based immersive Fe-C phase diagram with particle effects and interactive tooltips."""
    Ms_val = max(80.0, 539 - 423*C_val - 30.4*Mn - 12.1*Cr - 7.5*Mo)
    op_phase = get_phase(C_val, ht_temp)
    op_color = PHASE_COLORS.get(op_phase, "#ffffff")
    cfg = json_lib.dumps({
        "C": round(C_val, 4), "htT": round(ht_temp, 1), "tT": round(t_temp, 1),
        "Ms": round(Ms_val, 1), "proc": process_key, "cool": cool_medium,
        "opPh": op_phase, "opCol": op_color,
        "theme": {
            "isDark": IS_DARK,
            "panel": T['canvas_panel'], "text": T['canvas_text'],
            "grid": T['canvas_grid'], "border": T['border'],
            "fg": T['fg_primary'], "fgSec": T['fg_secondary'],
            "accent": T['accent'],
        },
    })
    return _IMMERSIVE_HTML.replace("__CFG__", cfg)


_IMMERSIVE_HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:transparent;overflow:hidden;font-family:Inter,-apple-system,sans-serif;color:var(--miq-text,#c0d0e8)}
#w{position:relative;width:100%;height:700px}
canvas{display:block;width:100%;height:100%;border-radius:8px;cursor:crosshair}
#tt{position:absolute;display:none;background:var(--miq-panel,rgba(6,10,22,0.94));backdrop-filter:blur(12px) saturate(140%);
border:1px solid var(--miq-border,rgba(160,185,220,0.22));border-radius:8px;padding:11px 15px;pointer-events:none;
z-index:10;min-width:200px;max-width:290px;box-shadow:0 8px 28px rgba(0,0,0,0.45)}
.tp{font-weight:700;font-size:14px;margin-bottom:4px}
.tc{font-size:11px;color:var(--miq-fgsec,rgba(170,190,220,0.7));font-family:monospace;margin-bottom:6px}
.td{font-size:11px;color:var(--miq-text,rgba(170,190,220,0.55));line-height:1.5}
#lg{position:absolute;bottom:6px;left:50%;transform:translateX(-50%);display:flex;gap:10px;
flex-wrap:wrap;justify-content:center;background:var(--miq-panel,rgba(6,10,22,0.82));
border:1px solid var(--miq-border,rgba(160,185,220,0.12));border-radius:6px;padding:5px 12px;z-index:5}
.li{display:flex;align-items:center;gap:4px;font-size:9px;color:var(--miq-fgsec,rgba(170,190,220,0.65));white-space:nowrap}
.ld{width:7px;height:7px;border-radius:50%;flex-shrink:0}
@media(max-width:600px){#w{height:480px}#lg{gap:6px;padding:4px 8px}.li{font-size:8px}}
</style></head><body>
<div id="w"><canvas id="c"></canvas>
<div id="tt"><div class="tp" id="tp"></div><div class="tc" id="tc"></div><div class="td" id="td"></div></div>
<div id="lg"></div></div>
<script>
var CFG=__CFG__;
(function(){var th=CFG.theme||{};var rs=document.documentElement.style;
rs.setProperty('--miq-panel',th.panel||'rgba(6,10,22,0.94)');
rs.setProperty('--miq-text',th.text||'rgba(170,190,220,0.55)');
rs.setProperty('--miq-fgsec',th.fgSec||'rgba(170,190,220,0.7)');
rs.setProperty('--miq-border',th.border||'rgba(160,185,220,0.22)');})();
var A1=723,A3F=912,PER=1495,MLT=1538,EUT=1147,CE=0.76,CA=2.11,MXC=2.3,MXT=1650,CP=0.17;
function a3(c){return A3F-(A3F-A1)/CE*Math.min(c,CE)}
function acm(c){return A1+(EUT-A1)/(CA-CE)*(c-CE)}
function lq(c){return c<=CP?MLT-(MLT-PER)/CP*c:c<=CA?PER+(EUT-PER)/(CA-CP)*(c-CP):EUT}
function ph(c,T){if(T>=lq(c))return'Liquid';
if(T>=A1){if(c<=CE&&T<a3(c))return'Austenite + Ferrite';if(c>CE&&T<acm(c))return'Austenite + Cementite';return'Austenite'}
if(c<=0.022)return'Ferrite';if(c<=CE)return'Ferrite + Pearlite';return'Pearlite + Cementite'}
var PI={'Liquid':{c:'#ffeaa0',d:'Molten steel \u2014 atoms in disordered liquid state. Maximum energy, no crystal structure. All elements fully dissolved.'},
'Austenite':{c:'#f0c830',d:'Face-centered cubic (FCC) \u03b3-iron. High-temperature stable phase. Dissolves up to 2.11% carbon. Non-magnetic. Parent phase for all transformations.'},
'Austenite + Ferrite':{c:'#e89030',d:'Intercritical region. FCC austenite and BCC ferrite coexist. Industrially used for dual-phase steel production with optimised strength-ductility.'},
'Austenite + Cementite':{c:'#d05548',d:'Hyper-eutectoid zone. Cementite (Fe\u2083C) precipitates from austenite along grain boundaries on slow cooling.'},
'Ferrite':{c:'#50c878',d:'Body-centered cubic (BCC) \u03b1-iron. Soft, ductile, magnetic. Maximum carbon solubility only 0.022 wt%. Matrix phase in low-carbon steels.'},
'Ferrite + Pearlite':{c:'#4aa8c0',d:'Two-phase equilibrium. Pearlite has lamellar ferrite + cementite structure. Balanced strength, hardness, and ductility.'},
'Pearlite + Cementite':{c:'#9a68c4',d:'Hyper-eutectoid mixture. Excess carbon forms continuous cementite network around pearlite colonies. Hard but brittle.'}};
var cv=document.getElementById('c'),cx=cv.getContext('2d'),wp=document.getElementById('w');
var W,H,ML=58,MR=40,MT=32,MB=52,PW,PH;
function rsz(){var r=wp.getBoundingClientRect();W=r.width;H=r.height;var d=window.devicePixelRatio||1;
cv.width=W*d;cv.height=H*d;cv.style.width=W+'px';cv.style.height=H+'px';
cx.setTransform(d,0,0,d,0,0);PW=W-ML-MR;PH=H-MT-MB}
rsz();window.addEventListener('resize',rsz);
function cX(c){return ML+c/MXC*PW}function tY(T){return MT+(1-T/MXT)*PH}
function xC(x){return Math.max(0,Math.min(MXC,(x-ML)/PW*MXC))}
function yT(y){return Math.max(0,Math.min(MXT,(1-(y-MT)/PH)*MXT))}
var ps=[];
function sp(x,y,t){if(ps.length>200)return;var p={x:x,y:y,a:0,t:t};
if(t===0){p.vx=(Math.random()-0.5)*0.7;p.vy=-(0.3+Math.random()*0.5);p.s=1.5+Math.random()*2;
p.l=55+Math.random()*55;p.c='255,'+(200+Math.random()*55|0)+','+(70+Math.random()*60|0)}
else if(t===1){p.vx=(Math.random()-0.5)*0.1;p.vy=(Math.random()-0.5)*0.1;p.s=1+Math.random();
p.l=80+Math.random()*70;p.c='240,200,48'}
else{p.vx=(Math.random()-0.5)*0.35;p.vy=(Math.random()-0.5)*0.35;p.s=1+Math.random()*1.3;
p.l=35+Math.random()*40;p.c=(200+Math.random()*55|0)+','+(110+Math.random()*40|0)+','+(20+Math.random()*30|0)}
ps.push(p)}
var tk=0,N=60;
function draw(){tk++;cx.clearRect(0,0,W,H);
var bg=cx.createLinearGradient(ML,H-MB,ML,MT);
bg.addColorStop(0,'#04060e');bg.addColorStop(0.18,'#080c18');bg.addColorStop(0.38,'#140a08');
bg.addColorStop(0.56,'#2a1006');bg.addColorStop(0.72,'#4a1a04');bg.addColorStop(0.86,'#803810');
bg.addColorStop(0.96,'#b85818');bg.addColorStop(1,'#d87828');
cx.fillStyle=bg;cx.fillRect(ML,MT,PW,PH);
cx.save();cx.beginPath();cx.rect(ML,MT,PW,PH);cx.clip();
var so=(tk*0.15)%6,i;
cx.fillStyle='rgba(80,200,120,0.07)';cx.beginPath();
cx.moveTo(cX(0),tY(0));cx.lineTo(cX(0.022),tY(0));cx.lineTo(cX(0.022),tY(A1));cx.lineTo(cX(0),tY(A1));cx.fill();
var gs=12;cx.fillStyle='rgba(80,200,120,0.10)';
for(var gx=cX(0);gx<cX(0.022);gx+=gs)for(var gy=tY(A1);gy<tY(0);gy+=gs){cx.beginPath();cx.arc(gx,gy,0.7,0,6.28);cx.fill()}
cx.fillStyle='rgba(74,168,192,0.08)';cx.beginPath();
cx.moveTo(cX(0.022),tY(0));cx.lineTo(cX(CE),tY(0));cx.lineTo(cX(CE),tY(A1));cx.lineTo(cX(0.022),tY(A1));cx.fill();
cx.strokeStyle='rgba(86,180,211,0.05)';cx.lineWidth=0.5;
for(var y=tY(A1);y<tY(0);y+=6){cx.beginPath();cx.moveTo(cX(0.022),y+so);cx.lineTo(cX(CE),y+so);cx.stroke()}
cx.fillStyle='rgba(154,104,196,0.08)';cx.beginPath();
cx.moveTo(cX(CE),tY(0));cx.lineTo(cX(MXC),tY(0));cx.lineTo(cX(MXC),tY(A1));cx.lineTo(cX(CE),tY(A1));cx.fill();
cx.strokeStyle='rgba(154,104,196,0.05)';
for(y=tY(A1);y<tY(0);y+=4){cx.beginPath();cx.moveTo(cX(CE),y+so);cx.lineTo(cX(MXC),y+so);cx.stroke()}
var tg=cx.createLinearGradient(0,tY(A3F),0,tY(A1));
tg.addColorStop(0,'rgba(232,144,48,0.04)');tg.addColorStop(1,'rgba(232,144,48,0.14)');
cx.fillStyle=tg;cx.beginPath();cx.moveTo(cX(0),tY(A3F));
for(i=0;i<=N;i++){var c=CE*i/N;cx.lineTo(cX(c),tY(a3(c)))}cx.lineTo(cX(CE),tY(A1));cx.lineTo(cX(0),tY(A1));cx.fill();
cx.fillStyle='rgba(208,85,72,0.08)';cx.beginPath();cx.moveTo(cX(CE),tY(A1));
for(i=0;i<=N;i++){c=CE+(CA-CE)*i/N;cx.lineTo(cX(c),tY(acm(c)))}cx.lineTo(cX(MXC),tY(EUT));cx.lineTo(cX(MXC),tY(A1));cx.fill();
var sh=0.07+0.03*Math.sin(tk*0.03);
cx.fillStyle='rgba(240,200,48,'+sh+')';cx.beginPath();cx.moveTo(cX(0),tY(MLT));cx.lineTo(cX(CP),tY(PER));
for(i=0;i<=N;i++){c=CE+(CA-CE)*i/N;cx.lineTo(cX(c),tY(acm(c)))}
cx.lineTo(cX(CE),tY(A1));for(i=N;i>=0;i--){c=CE*i/N;cx.lineTo(cX(c),tY(a3(c)))}cx.fill();
var lg2=0.07+0.025*Math.sin(tk*0.05);
cx.fillStyle='rgba(255,238,160,'+lg2+')';cx.beginPath();cx.moveTo(cX(0),tY(MXT));cx.lineTo(cX(MXC),tY(MXT));
for(i=N;i>=0;i--){c=MXC*i/N;cx.lineTo(cX(c),tY(lq(c)))}cx.fill();
var gw=1.6+0.25*Math.sin(tk*0.04);
cx.strokeStyle='#e0a030';cx.lineWidth=gw;cx.shadowColor='#e0a030';cx.shadowBlur=4;
cx.beginPath();for(i=0;i<=80;i++){c=CE*i/80;cx[i?'lineTo':'moveTo'](cX(c),tY(a3(c)))}cx.stroke();
cx.strokeStyle='#d05050';cx.shadowColor='#d05050';
cx.beginPath();for(i=0;i<=80;i++){c=CE+(CA-CE)*i/80;cx[i?'lineTo':'moveTo'](cX(c),tY(acm(c)))}cx.stroke();
cx.shadowBlur=0;cx.strokeStyle='rgba(128,170,238,0.55)';cx.lineWidth=1.4;cx.setLineDash([6,4]);
cx.beginPath();cx.moveTo(cX(0),tY(A1));cx.lineTo(cX(MXC),tY(A1));cx.stroke();cx.setLineDash([]);
cx.strokeStyle='rgba(255,240,160,0.65)';cx.shadowColor='#ffe8a0';cx.shadowBlur=5;cx.lineWidth=1.8;
cx.beginPath();for(i=0;i<=80;i++){c=MXC*i/80;cx[i?'lineTo':'moveTo'](cX(c),tY(lq(c)))}cx.stroke();
cx.shadowBlur=0;cx.strokeStyle='rgba(130,110,255,0.4)';cx.lineWidth=1.1;cx.setLineDash([7,3,2,3]);
cx.beginPath();cx.moveTo(cX(0),tY(CFG.Ms));cx.lineTo(cX(MXC),tY(CFG.Ms));cx.stroke();cx.setLineDash([]);
cx.strokeStyle='rgba(0,180,255,0.45)';cx.lineWidth=1.4;cx.setLineDash([7,5]);
cx.beginPath();cx.moveTo(cX(0),tY(CFG.htT));cx.lineTo(cX(MXC),tY(CFG.htT));cx.stroke();
if(CFG.tT>0){cx.strokeStyle='rgba(240,160,48,0.35)';cx.beginPath();cx.moveTo(cX(0),tY(CFG.tT));cx.lineTo(cX(MXC),tY(CFG.tT));cx.stroke()}
cx.strokeStyle='rgba(80,227,194,0.35)';cx.setLineDash([4,4]);
cx.beginPath();cx.moveTo(cX(CFG.C),tY(0));cx.lineTo(cX(CFG.C),tY(MXT));cx.stroke();cx.setLineDash([]);
var ox=cX(CFG.C),oy=tY(CFG.htT),pu=0.7+0.3*Math.sin(tk*0.06);
for(var r=26;r>=8;r-=4.5){cx.fillStyle='rgba(255,107,53,'+((1-r/26)*0.14*pu)+')';cx.beginPath();cx.arc(ox,oy,r,0,6.28);cx.fill()}
cx.fillStyle=CFG.opCol;cx.shadowColor=CFG.opCol;cx.shadowBlur=10;
cx.beginPath();cx.arc(ox,oy,5.5,0,6.28);cx.fill();
cx.strokeStyle='rgba(255,255,255,0.85)';cx.lineWidth=1.8;cx.shadowBlur=0;
cx.beginPath();cx.arc(ox,oy,5.5,0,6.28);cx.stroke();
if(tk%2===0)for(i=0;i<2;i++){c=Math.random()*MXC;var lt=lq(c);if(lt<MXT)sp(cX(c),tY(lt+Math.random()*(MXT-lt)),0)}
if(tk%5===0){c=Math.random()*CA;var lo=c<=CE?a3(c):acm(c),hi=lq(c);if(hi>lo&&hi>A1)sp(cX(c),tY(lo+Math.random()*(hi-lo)),1)}
if(tk%4===0){c=Math.random()*CE;var at=a3(c);if(at>A1)sp(cX(c),tY(A1+Math.random()*(at-A1)),2)}
for(i=ps.length-1;i>=0;i--){var p=ps[i];p.x+=p.vx;p.y+=p.vy;p.a++;
if(p.a>p.l||p.x<ML||p.x>W-MR||p.y<MT||p.y>H-MB){ps.splice(i,1);continue}
var al=(1-p.a/p.l)*0.45;cx.fillStyle='rgba('+p.c+','+al+')';cx.beginPath();cx.arc(p.x,p.y,p.s,0,6.28);cx.fill();
cx.fillStyle='rgba('+p.c+','+(al*0.2)+')';cx.beginPath();cx.arc(p.x,p.y,p.s*2.5,0,6.28);cx.fill()}
cx.restore();
cx.strokeStyle=(CFG.theme&&CFG.theme.grid)||'rgba(160,185,220,0.10)';cx.lineWidth=0.5;
for(var T=0;T<=MXT;T+=200){cx.beginPath();cx.moveTo(ML,tY(T));cx.lineTo(ML+PW,tY(T));cx.stroke()}
for(c=0;c<=MXC;c+=0.5){cx.beginPath();cx.moveTo(cX(c),MT);cx.lineTo(cX(c),MT+PH);cx.stroke()}
cx.fillStyle=(CFG.theme&&CFG.theme.text)||'rgba(160,185,220,0.62)';cx.font='10px monospace';cx.textAlign='right';
for(T=0;T<=MXT;T+=200)cx.fillText(T+'\u00b0C',ML-7,tY(T)+3);
cx.textAlign='center';for(c=0;c<=MXC;c+=0.5)cx.fillText(c.toFixed(1),cX(c),H-MB+16);
cx.fillStyle=(CFG.theme&&CFG.theme.fgSec)||'rgba(160,185,220,0.55)';cx.font='11px sans-serif';cx.textAlign='center';
cx.fillText('Carbon Content (wt%)',ML+PW/2,H-8);
cx.save();cx.translate(13,MT+PH/2);cx.rotate(-Math.PI/2);cx.fillText('Temperature (\u00b0C)',0,0);cx.restore();
cx.fillStyle=(CFG.theme&&CFG.theme.fg)||'rgba(220,235,255,0.55)';cx.font='bold 11px sans-serif';cx.textAlign='center';
var lbs=[[0.011,280,'\u03b1'],[0.38,300,'\u03b1+P'],[1.55,300,'P+Fe\u2083C'],
[0.25,790,'\u03b3+\u03b1'],[1.82,920,'\u03b3+Fe\u2083C'],[0.50,1080,'\u03b3 (Austenite)'],[1.1,1420,'Liquid']];
for(i=0;i<lbs.length;i++)cx.fillText(lbs[i][2],cX(lbs[i][0]),tY(lbs[i][1]));
cx.font='9px monospace';cx.textAlign='left';
cx.fillStyle='#e0a030';cx.fillText('A3',cX(0.03),tY(a3(0.03))-5);
cx.fillStyle='#d05050';cx.fillText('Acm',cX(CA-0.15),tY(acm(CA-0.15))-5);
cx.fillStyle='rgba(128,170,238,0.6)';cx.fillText('A1 = 723\u00b0C',cX(MXC)+3,tY(A1)+3);
cx.fillStyle='rgba(130,110,255,0.5)';cx.fillText('Ms = '+CFG.Ms.toFixed(0)+'\u00b0C',cX(MXC)+3,tY(CFG.Ms)+3);
cx.fillStyle='rgba(0,180,255,0.55)';cx.fillText(CFG.htT.toFixed(0)+'\u00b0C Aust.',cX(MXC)+3,tY(CFG.htT)+3);
if(CFG.tT>0){cx.fillStyle='rgba(240,160,48,0.5)';cx.fillText(CFG.tT.toFixed(0)+'\u00b0C Temper',cX(MXC)+3,tY(CFG.tT)+3)}
cx.fillStyle='rgba(80,227,194,0.45)';cx.textAlign='center';cx.fillText('C='+CFG.C.toFixed(2)+'%',cX(CFG.C),tY(MXT)-3);
cx.fillStyle='#ffd700';cx.save();cx.translate(cX(CE),tY(A1));cx.rotate(Math.PI/4);cx.fillRect(-3.5,-3.5,7,7);cx.restore();
cx.fillStyle='rgba(255,215,0,0.6)';cx.font='9px sans-serif';cx.textAlign='left';cx.fillText('S (Eutectoid)',cX(CE)+10,tY(A1)+3);
cx.fillStyle='#60c4ff';cx.beginPath();cx.moveTo(cX(CP),tY(PER)-4);cx.lineTo(cX(CP)-4,tY(PER)+4);cx.lineTo(cX(CP)+4,tY(PER)+4);cx.fill();
cx.fillStyle='rgba(96,196,255,0.6)';cx.fillText('P (Peritectic)',cX(CP)+10,tY(PER)+3);
cx.fillStyle='rgba(200,220,245,0.75)';cx.font='bold 13px Rajdhani,sans-serif';cx.textAlign='center';
cx.fillText('Fe\u2013C Phase Diagram  \u00b7  '+CFG.opPh,W/2,20);
cx.fillStyle='rgba(160,185,220,0.35)';cx.font='10px sans-serif';
cx.fillText('Hover to explore phases  \u00b7  C='+CFG.C.toFixed(2)+'%  T='+CFG.htT.toFixed(0)+'\u00b0C',W/2,MT-5);
cx.strokeStyle='rgba(160,185,220,0.12)';cx.lineWidth=1;cx.strokeRect(ML,MT,PW,PH);
requestAnimationFrame(draw)}draw();
var tt=document.getElementById('tt'),ttp=document.getElementById('tp'),ttc=document.getElementById('tc'),ttd=document.getElementById('td');
function showTT(mx,my,px,py){
if(mx<ML||mx>ML+PW||my<MT||my>MT+PH){tt.style.display='none';return}
var c2=xC(mx),T2=yT(my),p2=ph(c2,T2),info=PI[p2];if(!info){tt.style.display='none';return}
ttp.textContent=p2;ttp.style.color=info.c;
ttc.textContent='C = '+c2.toFixed(3)+' wt%  \u00b7  T = '+T2.toFixed(0)+' \u00b0C';ttd.textContent=info.d;
tt.style.display='block';var tx=px||mx+16,ty=py||my-8;
if(tx+240>W)tx=mx-250;if(ty+130>H)ty=my-130;if(ty<0)ty=4;
tt.style.left=tx+'px';tt.style.top=ty+'px'}
cv.addEventListener('mousemove',function(e){var r=cv.getBoundingClientRect();showTT(e.clientX-r.left,e.clientY-r.top)});
cv.addEventListener('mouseleave',function(){tt.style.display='none'});
cv.addEventListener('touchstart',function(e){e.preventDefault();var t=e.touches[0],r=cv.getBoundingClientRect();
showTT(t.clientX-r.left,t.clientY-r.top,8,8)},{passive:false});
cv.addEventListener('touchmove',function(e){e.preventDefault();var t=e.touches[0],r=cv.getBoundingClientRect();
showTT(t.clientX-r.left,t.clientY-r.top,8,8)},{passive:false});
cv.addEventListener('touchend',function(){tt.style.display='none'});
var lgd=document.getElementById('lg');var ks=Object.keys(PI);
for(i=0;i<ks.length;i++){var d=document.createElement('div');d.className='li';
d.innerHTML='<div class="ld" style="background:'+PI[ks[i]].c+'"></div>'+ks[i];lgd.appendChild(d)}
</script></body></html>"""


def _hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def build_tt_profile(process_key, ht_temp, t_temp, soak_time, t_time, C=0.35, Mn=0.0, Si=0.0, Ni=0.0, Cr=0.0, Mo=0.0):
    PROFILES = {
        "Quench_Temper": {
            "color": "#4c72b0", "label": "Quench & Temper",
            "stages": [
                ("Preheat",       "ramp", 0,  10,  25,     500),
                ("Heat to Aust",  "ramp", 10, 30,  500,    ht_temp),
                ("Soak",          "flat", 30, 30+soak_time/60, ht_temp, ht_temp),
                ("Quench",        "ramp", 30+soak_time/60, 35+soak_time/60, ht_temp, 50),
                ("Reheat temper", "ramp", 35+soak_time/60, 55+soak_time/60, 50, t_temp),
                ("Temper soak",   "flat", 55+soak_time/60, 55+soak_time/60+t_time/60, t_temp, t_temp),
                ("Air cool",      "ramp", 55+soak_time/60+t_time/60, 75+soak_time/60+t_time/60, t_temp, 25),
            ]
        },
        "Normalizing": {
            "color": "#50c878", "label": "Normalizing",
            "stages": [
                ("Heat",     "ramp", 0,  25,  25,     ht_temp),
                ("Soak",     "flat", 25, 25+soak_time/60, ht_temp, ht_temp),
                ("Air cool", "ramp", 25+soak_time/60, 75+soak_time/60, ht_temp, 25),
            ]
        },
        "Annealing": {
            "color": "#f0a030", "label": "Full Annealing",
            "stages": [
                ("Heat",         "ramp", 0,  25,  25,     ht_temp),
                ("Soak",         "flat", 25, 25+soak_time/60, ht_temp, ht_temp),
                ("Furnace cool", "ramp", 25+soak_time/60, 125+soak_time/60, ht_temp, 25),
            ]
        },
        "Stress_Relief": {
            "color": "#e05060", "label": "Stress Relief",
            "stages": [
                ("Heat",     "ramp", 0,  15,  25,     ht_temp),
                ("Soak",     "flat", 15, 15+soak_time/60, ht_temp, ht_temp),
                ("Air cool", "ramp", 15+soak_time/60, 45+soak_time/60, ht_temp, 25),
            ]
        },
    }
    prof   = PROFILES.get(process_key, PROFILES["Quench_Temper"])
    color  = prof["color"]
    times, temps = [0], [25]
    for _, _type, t1, t2, T1, T2 in prof["stages"]:
        times.append(t2); temps.append(T2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=temps, mode="lines",
        line=dict(color=color, width=3),
        fill="tozeroy", fillcolor=_hex_to_rgba(color, 0.12),
        hovertemplate="t = %{x:.1f} min<br>T = %{y:.0f} C<extra></extra>",
        name=prof["label"],
    ))
    A1_val = 723
    A3_val = calc_a3(C, Mn=Mn, Si=Si, Ni=Ni, Cr=Cr, Mo=Mo)
    fig.add_trace(go.Scatter(
        x=[0, max(times)], y=[A1_val, A1_val],
        line=dict(color="#80aaee", width=1, dash="dot"),
        mode="lines+text",
        text=["", f"A1 = {A1_val}C"],
        textposition="middle right", textfont=dict(size=9, color="#80aaee"),
        name="A1", showlegend=False))
    if ht_temp > A3_val:
        fig.add_trace(go.Scatter(
            x=[0, max(times)], y=[A3_val, A3_val],
            line=dict(color="#f0a030", width=1, dash="dot"),
            mode="lines+text",
            text=["", f"A3 = {A3_val:.0f}C"],
            textposition="middle right", textfont=dict(size=9, color="#f0a030"),
            name="A3", showlegend=False))
    fig.update_layout(
        **_BASE,
        title=dict(text=f"<b>Temperature-Time Profile</b>  - {prof['label']}",
                   font=dict(size=13), x=0.5, xanchor="center"),
        xaxis=dict(title="Time (min)", gridcolor="rgba(180,210,255,0.06)"),
        yaxis=dict(title="Temperature (C)", range=[0, max(ht_temp*1.12, 250)],
                   gridcolor="rgba(180,210,255,0.06)"),
        height=370, margin=dict(l=55, r=80, t=50, b=50),
    )
    return fig


def build_cct_diagram(C, Mn=0.85, Si=0.25, Ni=0.20, Cr=1.05, Mo=0.20,
                      process_key="Quench_Temper", cool_medium="Oil",
                      ht_temp=850, ASTM_gs=8):
    """
    Approximate CCT (Continuous Cooling Transformation) diagram.
    Uses simplified Kirkaldy–Venugopalan empirical model for incubation times
    of ferrite, pearlite, and bainite transformations as a function of
    undercooling below Ae3/Ae1, grain size, and composition.
    """
    # Equilibrium temperatures (Andrews 1965)
    Ae3 = 912.0 - 203.0*np.sqrt(max(C, 1e-6)) - 30.0*Mn + 44.7*Si - 15.2*Ni + 31.5*Mo
    Ae1 = 723.0 - 10.7*Mn + 29.1*Si + 16.9*Cr - 16.9*Ni
    Ae1 = max(Ae1, 680.0)
    Ms  = max(50.0, 539.0 - 423*C - 30.4*Mn - 12.1*Cr - 7.5*Mo - 7.5*Si)
    Bs  = max(Ms + 50, 830.0 - 270*C - 90*Mn - 37*Ni - 70*Cr - 83*Mo)

    D_gamma = 2**(-(ASTM_gs - 1) / 2.0)  # Prior austenite grain diameter in mm

    # Simplified incubation time model (log10 seconds) based on Kirkaldy
    # Ferrite start: C-curve shape with nose around Ae3-100
    def tau_ferrite(T):
        dT = max(Ae3 - T, 1.0)
        if T > Ae3 or T < Ae1:
            return 1e8
        x = dT / max(Ae3 - Ae1, 1.0)
        # Nose at x ~ 0.5, deepening with hardenability
        alloy_factor = (1 + 1.0*Mn) * (1 + 0.7*Cr) * (1 + 0.5*Mo) * (1 + 0.3*Ni)
        tau_nose = 0.5 * alloy_factor * D_gamma**0.5
        shape = np.exp(3.5 * (x - 0.45)**2)
        return tau_nose * shape

    # Pearlite start
    def tau_pearlite(T):
        if T > Ae1 or T < 400:
            return 1e8
        dT = max(Ae1 - T, 1.0)
        x = dT / max(Ae1 - 400, 1.0)
        alloy_factor = (1 + 1.5*Mn) * (1 + 1.2*Cr) * (1 + 2.0*Mo) * (1 + 0.5*Ni)
        tau_nose = 2.0 * alloy_factor * D_gamma**0.5
        shape = np.exp(4.0 * (x - 0.35)**2)
        return max(0.5, tau_nose * shape)

    # Bainite start
    def tau_bainite(T):
        if T > Bs or T < Ms:
            return 1e8
        dT = max(Bs - T, 1.0)
        x = dT / max(Bs - Ms, 1.0)
        alloy_factor = (1 + 1.8*C) * (1 + 0.6*Mn) * (1 + 0.8*Cr) * (1 + 1.5*Mo)
        tau_nose = 5.0 * alloy_factor
        shape = np.exp(3.0 * (x - 0.40)**2)
        return max(1.0, tau_nose * shape)

    # Generate C-curves
    T_ferrite = np.linspace(Ae3 - 5, Ae1 + 5, 80)
    t_ferrite = np.array([tau_ferrite(T) for T in T_ferrite])
    valid_f = t_ferrite < 1e6
    T_ferrite, t_ferrite = T_ferrite[valid_f], t_ferrite[valid_f]

    T_pearlite = np.linspace(Ae1 - 5, max(420, Ms + 30), 80)
    t_pearlite = np.array([tau_pearlite(T) for T in T_pearlite])
    valid_p = t_pearlite < 1e6
    T_pearlite, t_pearlite = T_pearlite[valid_p], t_pearlite[valid_p]

    T_bainite = np.linspace(Bs - 5, Ms + 10, 80)
    t_bainite = np.array([tau_bainite(T) for T in T_bainite])
    valid_b = t_bainite < 1e6
    T_bainite, t_bainite = T_bainite[valid_b], t_bainite[valid_b]

    fig = go.Figure()

    # Ferrite start curve
    if len(t_ferrite) > 2:
        fig.add_trace(go.Scatter(
            x=t_ferrite, y=T_ferrite,
            mode="lines", name="Ferrite start",
            line=dict(color="#50c878", width=2.5),
            hovertemplate="Ferrite start<br>t = %{x:.1f} s<br>T = %{y:.0f} C<extra></extra>",
        ))

    # Pearlite start curve
    if len(t_pearlite) > 2:
        fig.add_trace(go.Scatter(
            x=t_pearlite, y=T_pearlite,
            mode="lines", name="Pearlite start",
            line=dict(color="#c9a56b", width=2.5),
            hovertemplate="Pearlite start<br>t = %{x:.1f} s<br>T = %{y:.0f} C<extra></extra>",
        ))

    # Bainite start curve
    if len(t_bainite) > 2:
        fig.add_trace(go.Scatter(
            x=t_bainite, y=T_bainite,
            mode="lines", name="Bainite start",
            line=dict(color="#8172b3", width=2.5),
            hovertemplate="Bainite start<br>t = %{x:.1f} s<br>T = %{y:.0f} C<extra></extra>",
        ))

    # Ms and Bs horizontal lines
    x_range = [0.1, 1e5]
    fig.add_trace(go.Scatter(
        x=x_range, y=[Ms, Ms],
        mode="lines", name=f"Ms = {Ms:.0f} C",
        line=dict(color="#6faed9", width=1.5, dash="dashdot"),
    ))
    fig.add_trace(go.Scatter(
        x=x_range, y=[Bs, Bs],
        mode="lines", name=f"Bs = {Bs:.0f} C",
        line=dict(color="#9370db", width=1.3, dash="dashdot"),
    ))
    fig.add_trace(go.Scatter(
        x=x_range, y=[Ae3, Ae3],
        mode="lines", name=f"Ae3 = {Ae3:.0f} C",
        line=dict(color="#e0a030", width=1.2, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=x_range, y=[Ae1, Ae1],
        mode="lines", name=f"Ae1 = {Ae1:.0f} C",
        line=dict(color="#80aaee", width=1.2, dash="dot"),
    ))

    # Cooling curves for different media
    COOL_RATES = {
        "Water": (80.0, "#60c4ff"),
        "Oil": (25.0, "#f0a030"),
        "Polymer": (35.0, "#c860ff"),
        "Salt Bath": (15.0, "#ffd700"),
        "Air": (2.0, "#50e3c2"),
        "Furnace": (0.1, "#ff6b35"),
    }
    for medium, (rate, color) in COOL_RATES.items():
        # Simplified exponential cooling: T(t) = T_aust * exp(-rate/300 * t)
        t_vals = np.logspace(-1, 5, 200)
        T_vals = ht_temp * np.exp(-rate / 300.0 * t_vals)
        T_vals = np.clip(T_vals, 20, ht_temp)
        is_selected = (medium == cool_medium)
        fig.add_trace(go.Scatter(
            x=t_vals, y=T_vals,
            mode="lines",
            name=f"Cool: {medium}" + (" (selected)" if is_selected else ""),
            line=dict(color=color, width=2.5 if is_selected else 1.0,
                      dash="solid" if is_selected else "dash"),
            opacity=1.0 if is_selected else 0.35,
            hovertemplate=f"{medium}<br>t = %{{x:.1f}} s<br>T = %{{y:.0f}} C<extra></extra>",
        ))

    # Region labels
    for x, y, txt in [
        (0.8, (Ae3 + Ae1) / 2, "FERRITE"),
        (3.0, (Ae1 + max(420, Ms + 30)) / 2, "PEARLITE"),
        (8.0, (Bs + Ms) / 2, "BAINITE"),
        (0.3, Ms - 40, "MARTENSITE"),
    ]:
        fig.add_annotation(
            x=np.log10(x), y=y, text=f"<b>{txt}</b>",
            showarrow=False, font=dict(size=10, color="rgba(200,220,245,0.55)"),
            xref="x", yref="y",
        )

    fig.update_layout(
        **_BASE,
        title=dict(
            text=(f"<b>CCT Diagram</b>  ·  "
                  f"C={C:.2f}  Mn={Mn:.2f}  Cr={Cr:.2f}  Mo={Mo:.2f}  "
                  f"(ASTM GS {ASTM_gs})"),
            font=dict(size=12), x=0.5, xanchor="center",
        ),
        xaxis=dict(
            title="Time (seconds)",
            type="log",
            range=[-1, 5],
            gridcolor="rgba(180,210,255,0.06)",
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            title="Temperature (C)",
            range=[0, max(ht_temp + 50, Ae3 + 60)],
            gridcolor="rgba(180,210,255,0.06)",
            tickfont=dict(size=10),
        ),
        legend=dict(
            bgcolor="rgba(4,10,24,0.88)",
            bordercolor="rgba(61,184,255,0.18)",
            borderwidth=1,
            font=dict(size=9, color="#c0d8f0"),
            x=1.02, y=1, xanchor="left",
        ),
        height=560,
        margin=dict(l=58, r=180, t=55, b=55),
    )
    return fig


def steel_grade(tensile):
    if tensile < 600:   return "Low Strength",        "#56b4d3"
    if tensile < 900:   return "Medium Strength",     "#50c878"
    if tensile < 1200:  return "High Strength",       "#ffd700"
    if tensile < 1500:  return "Very High Strength",  "#ff8c30"
    return "Ultra-High Strength", "#ff5c6a"


# ══════════════════════════════════════════════════════════════════════════════
#  MICROSTRUCTURE — VORONOI HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _compute_voronoi_grains(n_grains, seed=42):
    from scipy.spatial import Voronoi
    rng  = np.random.default_rng(seed)
    pts  = rng.random((n_grains, 2))
    offs = np.array([[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]])
    all_pts = np.vstack([pts] + [pts + o for o in offs])
    vor  = Voronoi(all_pts)
    polys = []
    for i in range(n_grains):
        reg = vor.regions[vor.point_region[i]]
        if -1 in reg or len(reg) < 3:
            polys.append(None)
        else:
            polys.append(np.clip(vor.vertices[reg], 0.0, 1.0))
    return pts, polys


# ══════════════════════════════════════════════════════════════════════════════
#  COMBINED PLOTLY ANIMATION
#  Left  : Fe-C diagram with animated operating point tracking T-t path
#  Right : EBSD-style microstructure image (updated each frame via layout.images)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def build_immersive_simulation_html(process_key, C, ht_temp, soak_time, cool_medium,
                                     t_temp, t_time, theme_key="dark"):
    """Canvas-based immersive simulation: phase diagram + microstructure evolution."""
    from PIL import Image as PILImage
    from io import BytesIO
    from scipy.spatial import cKDTree

    A1    = 723.0
    Ms    = max(80.0, 539 - 423*C - 30.4*0.85 - 12.1*1.05 - 7.5*0.2)
    COOL  = {"Water":3.0,"Polymer":2.2,"Oil":1.8,"Salt Bath":1.1,"Air":0.4,"Furnace":0.08}
    spd   = COOL.get(cool_medium, 1.0)
    K_gg  = 0.9 * np.exp(-20000 / (ht_temp + 273.15)) * (soak_time / 60.0)
    n_ini = 100
    n_soak= max(16, int(n_ini / (1.0 + 7.0 * K_gg)))

    PC = {
        "ferrite":             np.array([0.31, 0.78, 0.47]),
        "pearlite":            np.array([0.52, 0.35, 0.20]),
        "austenite":           np.array([1.00, 0.84, 0.00]),
        "martensite":          np.array([0.10, 0.14, 0.55]),
        "tempered_martensite": np.array([0.25, 0.42, 0.90]),
        "bainite":             np.array([0.42, 0.22, 0.65]),
        "fine_pearlite":       np.array([0.58, 0.38, 0.18]),
        "coarse_pearlite":     np.array([0.72, 0.52, 0.28]),
    }
    PCLABELS = {
        "ferrite":"Ferrite", "pearlite":"Pearlite", "austenite":"Austenite",
        "martensite":"Martensite", "tempered_martensite":"Tempered Mart.",
        "bainite":"Bainite", "fine_pearlite":"Fine Pearlite", "coarse_pearlite":"Coarse Pearlite",
    }
    PCCOLORS = {
        "ferrite":"#50c878", "pearlite":"#855930", "austenite":"#f0d600",
        "martensite":"#1a238c", "tempered_martensite":"#406be6",
        "bainite":"#6c38a6", "fine_pearlite":"#946020", "coarse_pearlite":"#b88548",
    }

    seed0 = (int(abs(C * 100)) + int(ht_temp)) % 9973
    GS = {
        "initial": _compute_voronoi_grains(n_ini, seed=seed0),
        "soaked":  _compute_voronoi_grains(n_soak, seed=seed0+1),
        "recryst": _compute_voronoi_grains(n_soak+28, seed=seed0+2),
    }
    rng0 = np.random.default_rng(seed0)
    ORI  = {k: rng0.random(len(v[1])) for k, v in GS.items()}

    IMG  = 160
    gx, gy = np.meshgrid(np.linspace(0, 1, IMG), np.linspace(0, 1, IMG))
    pixels = np.column_stack([gx.ravel(), gy.ravel()])
    GIDX = {}
    for k, (pts, _) in GS.items():
        _, idx = cKDTree(pts).query(pixels)
        GIDX[k] = idx.clip(0, len(pts)-1)

    def T_at(f):
        if process_key == "Quench_Temper":
            if f < 0.18: return 25 + (ht_temp-25)*f/0.18
            if f < 0.38: return float(ht_temp)
            if f < 0.52: return ht_temp - (ht_temp-50)*(f-0.38)/0.14
            if f < 0.65: return 50 + (max(t_temp,50)-50)*(f-0.52)/0.13
            if f < 0.88: return float(max(t_temp, 50))
            return max(t_temp,50) - (max(t_temp,50)-25)*(f-0.88)/0.12
        elif process_key == "Normalizing":
            if f < 0.22: return 25 + (ht_temp-25)*f/0.22
            if f < 0.42: return float(ht_temp)
            return ht_temp - (ht_temp-25)*(f-0.42)/0.58
        elif process_key == "Annealing":
            if f < 0.20: return 25 + (ht_temp-25)*f/0.20
            if f < 0.45: return float(ht_temp)
            return ht_temp - (ht_temp-25)*(f-0.45)/0.55
        else:
            if f < 0.22: return 25 + (ht_temp-25)*f/0.22
            if f < 0.68: return float(ht_temp)
            return ht_temp - (ht_temp-25)*(f-0.68)/0.32

    def sname(f):
        if process_key == "Quench_Temper":
            if f < 0.18: return "Heating"
            if f < 0.38: return "Austenitizing"
            if f < 0.52: return "Quenching"
            if f < 0.65: return "Re-heating"
            if f < 0.88: return "Tempering"
            return "Air Cooling"
        elif process_key == "Normalizing":
            if f < 0.22: return "Heating to Austenite"
            if f < 0.42: return "Austenitizing (Grain Refinement)"
            prog = (f - 0.42) / 0.58
            if prog < 0.4: return "Air Cooling (Recrystallization)"
            return "Air Cooling (Pearlite Formation)"
        elif process_key == "Annealing":
            if f < 0.20: return "Heating to Austenite"
            if f < 0.45: return "Austenitizing (Full Dissolution)"
            prog = (f - 0.45) / 0.55
            if prog < 0.3: return "Furnace Cooling (Slow)"
            return "Furnace Cooling (Coarse Pearlite)"
        else:
            if f < 0.22: return "Heating (Sub-A1)"
            if f < 0.68: return "Stress Relief Soaking"
            return "Air Cooling (Structure Retained)"

    def frame_state(f):
        T = T_at(f)
        if process_key == "Quench_Temper":
            if f < 0.18:
                af = max(0.0, min(1.0, (T-A1)/max(1,ht_temp-A1)))*(f/0.18)
                return "initial",[("austenite",af),("ferrite",max(0,0.55-af*0.5)),("pearlite",max(0,0.45-af*0.5))],10,0.0
            if f < 0.38:
                gs = "soaked" if (f-0.18)/0.20>0.5 else "initial"
                return gs,[("austenite",1.0)],20,0.0
            if f < 0.52:
                prog = (f-0.38)/0.14
                if spd >= 1.5:  # Fast quench (water, polymer, oil) → martensite
                    mf = max(0.0, min(1.0, (Ms+60-T)/(Ms+60-30)))
                    return "soaked",[("martensite",mf),("austenite",1-mf)],30,mf
                elif spd >= 0.3:  # Medium quench (air, salt bath) → bainite
                    bf = max(0.0, min(0.7, prog*0.7))
                    mf = max(0.0, min(0.3, prog*0.3))
                    return "soaked",[("bainite",bf),("martensite",mf),("austenite",max(0,1-bf-mf))],30,mf
                else:  # Very slow (furnace) → pearlite
                    pf = max(0.0, min(0.65, prog*0.65))
                    ff = max(0.0, min(0.35, prog*0.35))
                    return "soaked",[("fine_pearlite",pf),("ferrite",ff),("austenite",max(0,1-pf-ff))],30,0.0
            if f < 0.65:
                if spd >= 1.5:
                    return "soaked",[("martensite",1.0)],35,0.8
                elif spd >= 0.3:
                    return "soaked",[("bainite",0.7),("martensite",0.3)],35,0.3
                else:
                    return "soaked",[("fine_pearlite",0.65),("ferrite",0.35)],35,0.0
            if f < 0.88:
                tf = min(1.0,(f-0.65)/0.23)
                return "soaked",[("tempered_martensite",tf),("martensite",1-tf)],40,max(0,0.5-tf*0.5)
            return "soaked",[("tempered_martensite",1.0)],45,0.0
        elif process_key == "Normalizing":
            if f < 0.22:
                af = max(0.0,min(1.0,(T-A1)/max(1,ht_temp-A1)))*(f/0.22)
                return "initial",[("austenite",af),("ferrite",max(0,0.55*(1-af))),("pearlite",max(0,0.45*(1-af)))],10,0.0
            if f < 0.42:
                gs = "soaked" if (f-0.22)/0.20>0.5 else "initial"
                return gs,[("austenite",1.0)],20,0.0
            prog = min(1.0,(f-0.42)/0.58)
            X_rx = 1-np.exp(-2.0*prog**2.0)
            gs = "recryst" if X_rx>0.45 else "soaked"
            fp = min(0.65,prog*0.65); fe=min(0.35,prog*0.35)
            return gs,[("fine_pearlite",fp),("ferrite",fe),("austenite",max(0,1-fp-fe))],50,0.0
        elif process_key == "Annealing":
            if f < 0.20:
                af = max(0.0,min(1.0,(T-A1)/max(1,ht_temp-A1)))*(f/0.20)
                return "initial",[("austenite",af),("ferrite",max(0,0.55*(1-af))),("pearlite",max(0,0.45*(1-af)))],10,0.0
            if f < 0.45:
                gs = "soaked" if (f-0.20)/0.25>0.5 else "initial"
                return gs,[("austenite",1.0)],20,0.0
            prog = min(1.0,(f-0.45)/0.55)
            X_rx = 1-np.exp(-1.5*prog**1.8)
            gs = "recryst" if X_rx>0.40 else "soaked"
            cp = min(0.68,prog*0.68); fe=min(0.30,prog*0.30)
            return gs,[("coarse_pearlite",cp),("ferrite",fe),("austenite",max(0,1-cp-fe))],60,0.0
        else:  # Stress Relief
            # No phase change below A1 — but show progressive stress relaxation
            # via subtle grain structure smoothing (seed_off changes visual texture)
            if f < 0.22:
                # Heating — no phase change, just warming
                prog = f / 0.22
                return "initial",[("ferrite",0.55),("pearlite",0.45)],int(70 + prog*5),0.0
            if f < 0.68:
                # Soaking at sub-A1 — recovery and dislocation annihilation
                prog = (f - 0.22) / 0.46
                # Slight shift in phase fractions to visually indicate recovery
                return "initial",[("ferrite",0.55 + prog*0.03),("pearlite",0.45 - prog*0.03)],int(78 + prog*8),0.0
            # Cooling — structure retained but "relaxed"
            prog = (f - 0.68) / 0.32
            return "initial",[("ferrite",0.58),("pearlite",0.42)],int(88 + prog*4),0.0

    def make_micro_array(gs_key, phase_fracs, seed_off, needle_frac):
        pts, _ = GS[gs_key]
        oris   = ORI[gs_key]
        n_g    = len(pts)
        rng_l  = np.random.default_rng(seed0 + seed_off)
        cum    = np.cumsum([pf for _, pf in phase_fracs])
        cum    = cum / max(cum[-1], 1e-9)
        g_rgb  = np.zeros((n_g, 3))
        for i in range(n_g):
            r   = rng_l.random()
            idx = min(int(np.searchsorted(cum, r)), len(phase_fracs)-1)
            lbl = phase_fracs[idx][0]
            b   = 0.58 + 0.42 * oris[min(i, len(oris)-1)]
            g_rgb[i] = np.clip(PC.get(lbl, PC["ferrite"]) * b, 0, 1)
        gidx   = GIDX[gs_key]
        px_rgb = g_rgb[gidx].reshape(IMG, IMG, 3)
        gmap = gidx.reshape(IMG, IMG)
        bnd  = np.zeros((IMG, IMG), dtype=bool)
        bnd[:-1]   |= (gmap[:-1]   != gmap[1:])
        bnd[:, :-1] |= (gmap[:, :-1] != gmap[:, 1:])
        px_rgb[bnd] *= 0.14
        if needle_frac > 0.1:
            rng_n = np.random.default_rng(seed0 + int(needle_frac * 300))
            for _ in range(int(55 * needle_frac)):
                x0, y0 = rng_n.random(), rng_n.random()
                ang    = rng_n.random() * np.pi
                L      = 0.04 + rng_n.random() * 0.09
                n_pts  = max(4, int(L * IMG * 2))
                ts     = np.linspace(0, L, n_pts)
                xs = np.clip((x0+ts*np.cos(ang))*IMG, 0, IMG-1).astype(int)
                ys = np.clip((y0+ts*np.sin(ang))*IMG, 0, IMG-1).astype(int)
                px_rgb[ys, xs] = [0.70, 0.87, 1.0]
        return (px_rgb * 255).clip(0, 255).astype(np.uint8)

    def arr_to_b64(arr):
        pil = PILImage.fromarray(arr)
        buf = BytesIO()
        pil.save(buf, format='PNG', optimize=True)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    # Build frame data
    N_FRAMES = 32
    fracs = np.linspace(0, 1, N_FRAMES)
    frames_json = []
    for fi, f in enumerate(fracs):
        T = T_at(f)
        stage = sname(f)
        gs_key, phase_fracs, seed_off, needle_frac = frame_state(f)
        arr = make_micro_array(gs_key, phase_fracs, seed_off, needle_frac)
        img_src = arr_to_b64(arr)
        n_vis = sum(1 for p in GS[gs_key][1] if p is not None)
        phases = []
        for lbl, frac in phase_fracs:
            if frac > 0.01:
                phases.append({"n": PCLABELS.get(lbl, lbl), "f": round(frac, 3),
                               "c": PCCOLORS.get(lbl, "#888")})
        frames_json.append({
            "T": round(T, 1), "s": stage, "img": img_src,
            "g": n_vis, "ph": phases,
        })

    # Build T-t path for the mini chart
    path_pts = [{"f": round(f, 4), "T": round(T_at(f), 1)} for f in np.linspace(0, 1, 80)]

    th = THEMES.get(theme_key, THEMES["dark"])
    cfg = json_lib.dumps({
        "C": round(C, 4), "htT": round(ht_temp, 1), "tT": round(t_temp, 1),
        "Ms": round(Ms, 1), "proc": process_key, "cool": cool_medium,
        "frames": frames_json, "path": path_pts,
        "theme": {
            "isDark": (theme_key == "dark"),
            "panel":  th['canvas_panel'], "text": th['canvas_text'],
            "grid":   th['canvas_grid'],  "border": th['border'],
            "fg":     th['fg_primary'],   "fgSec": th['fg_secondary'],
            "accent": th['accent'],       "stage": th['accent_cool'],
            "tval":   th['accent'],
        },
    })
    return _SIM_HTML.replace("__CFG__", cfg)


_SIM_HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:transparent;overflow:hidden;font-family:Inter,-apple-system,sans-serif;color:var(--miq-text,#c0d0e8)}
#w{position:relative;width:100%;height:780px;display:flex;flex-direction:column}
#top{display:flex;align-items:center;justify-content:space-between;padding:10px 16px;
background:var(--miq-panel,rgba(6,10,22,0.90));border:1px solid var(--miq-border,rgba(160,185,220,0.12));border-radius:8px;margin-bottom:6px}
#stage{font-weight:700;font-size:15px;color:var(--miq-stage,#c9a56b);font-family:Rajdhani,sans-serif;letter-spacing:0.08em;text-transform:uppercase}
#tval{font-size:22px;font-weight:700;font-family:'JetBrains Mono',monospace;color:var(--miq-tval,#e08030)}
#tunit{font-size:11px;color:var(--miq-fgsec,rgba(170,190,220,0.6));margin-left:2px}
#panels{display:flex;flex:1;gap:6px;min-height:0}
.panel{flex:1;background:var(--miq-panel,rgba(6,10,22,0.60));border:1px solid var(--miq-border,rgba(160,185,220,0.10));
border-radius:8px;position:relative;overflow:hidden;display:flex;flex-direction:column}
.ptitle{font-size:11px;font-weight:600;color:var(--miq-fgsec,rgba(170,190,220,0.65));text-transform:uppercase;
letter-spacing:0.12em;padding:8px 12px 4px;font-family:Rajdhani,sans-serif}
canvas,#mimg{display:block;width:100%;flex:1;min-height:0;object-fit:contain;border-radius:0 0 7px 7px}
#phbar{display:flex;height:18px;border-radius:4px;overflow:hidden;margin:0 10px 8px}
.phseg{height:100%;transition:width 0.3s}
#phleg{display:flex;gap:8px;flex-wrap:wrap;padding:0 10px 6px}
.phl{display:flex;align-items:center;gap:3px;font-size:9px;color:var(--miq-fgsec,rgba(170,190,220,0.65))}
.phd{width:6px;height:6px;border-radius:50%}
#ctrls{display:flex;align-items:center;gap:10px;padding:10px 16px;margin-top:6px;
background:var(--miq-panel,rgba(6,10,22,0.90));border:1px solid var(--miq-border,rgba(160,185,220,0.12));border-radius:8px}
#pbtn{width:38px;height:38px;border-radius:50%;border:1px solid var(--miq-stage,rgba(201,165,107,0.5));
background:transparent;color:var(--miq-stage,#c9a56b);font-size:16px;cursor:pointer;display:flex;align-items:center;
justify-content:center;transition:all 0.2s;flex-shrink:0}
#pbtn:hover{background:rgba(201,165,107,0.10);border-color:var(--miq-stage,#c9a56b)}
#sbar{flex:1;position:relative;height:28px;display:flex;align-items:center;cursor:pointer}
#strack{width:100%;height:4px;background:var(--miq-grid,rgba(160,185,220,0.10));border-radius:2px;position:relative}
#sfill{height:100%;background:linear-gradient(90deg,var(--miq-stage,#c9a56b),var(--miq-tval,#e08030));border-radius:2px;width:0%;transition:width 0.1s}
#sthumb{position:absolute;top:50%;width:14px;height:14px;border-radius:50%;background:var(--miq-fg,#e4e9f2);
border:2px solid var(--miq-stage,#c9a56b);transform:translate(-50%,-50%);left:0%;transition:left 0.1s;
box-shadow:0 0 0 3px rgba(201,165,107,0.18)}
#ftime{font-size:11px;color:var(--miq-fgsec,rgba(170,190,220,0.55));font-family:monospace;min-width:55px;text-align:right;flex-shrink:0}
@media(max-width:700px){#panels{flex-direction:column}#w{height:auto;min-height:700px}
.panel{min-height:260px}#tval{font-size:18px}#stage{font-size:12px}}
</style></head><body>
<div id="w">
<div id="top"><div id="stage">Heating</div><div><span id="tval">25</span><span id="tunit">&deg;C</span></div></div>
<div id="panels">
<div class="panel"><div class="ptitle">Fe&ndash;C Phase Diagram</div><canvas id="pc"></canvas></div>
<div class="panel"><div class="ptitle">Microstructure (EBSD)</div><img id="mimg" src="" alt="micro">
<div id="phbar"></div><div id="phleg"></div></div>
</div>
<div id="ctrls">
<button id="pbtn">&#9654;</button>
<div id="sbar"><div id="strack"><div id="sfill"></div><div id="sthumb"></div></div></div>
<div id="ftime">1 / 32</div>
</div>
</div>
<script>
var D=__CFG__;
(function(){var th=D.theme||{};var rs=document.documentElement.style;
rs.setProperty('--miq-panel',th.panel||'rgba(6,10,22,0.90)');
rs.setProperty('--miq-text',th.text||'rgba(170,190,220,0.62)');
rs.setProperty('--miq-fgsec',th.fgSec||'rgba(170,190,220,0.7)');
rs.setProperty('--miq-grid',th.grid||'rgba(160,185,220,0.10)');
rs.setProperty('--miq-border',th.border||'rgba(160,185,220,0.12)');
rs.setProperty('--miq-fg',th.fg||'#e4e9f2');
rs.setProperty('--miq-stage',th.stage||'#c9a56b');
rs.setProperty('--miq-tval',th.tval||'#e08030');
rs.setProperty('--miq-accent',th.accent||'#c9a56b');})();
var F=D.frames,NF=F.length,ci=0,playing=false,timer=null;
var A1=723,A3F=912,PER=1495,MLT=1538,EUT=1147,CE=0.76,CA=2.11,MXC=2.3,MXT=1650,CP=0.17;
function a3(c){return A3F-(A3F-A1)/CE*Math.min(c,CE)}
function acm(c){return A1+(EUT-A1)/(CA-CE)*(c-CE)}
function lq(c){return c<=CP?MLT-(MLT-PER)/CP*c:c<=CA?PER+(EUT-PER)/(CA-CP)*(c-CP):EUT}
var cv=document.getElementById('pc'),cx=cv.getContext('2d');
var mimg=document.getElementById('mimg');
var stageEl=document.getElementById('stage'),tvalEl=document.getElementById('tval');
var phbar=document.getElementById('phbar'),phleg=document.getElementById('phleg');
var pbtn=document.getElementById('pbtn'),sfill=document.getElementById('sfill');
var sthumb=document.getElementById('sthumb'),ftime=document.getElementById('ftime');
var sbar=document.getElementById('sbar');
function rsz(){
var p=cv.parentElement;var r=p.getBoundingClientRect();
var w=r.width,h=r.height-30;
var d=window.devicePixelRatio||1;
cv.width=w*d;cv.height=h*d;cv.style.width=w+'px';cv.style.height=h+'px';
cx.setTransform(d,0,0,d,0,0);drawPhase()}
rsz();window.addEventListener('resize',rsz);
function drawPhase(){
var W=cv.width/(window.devicePixelRatio||1),H=cv.height/(window.devicePixelRatio||1);
var ML=44,MR=28,MT=10,MB=32,PW=W-ML-MR,PH=H-MT-MB;
function cX(c){return ML+c/MXC*PW}function tYf(T){return MT+(1-T/MXT)*PH}
cx.clearRect(0,0,W,H);
var bg=cx.createLinearGradient(ML,H-MB,ML,MT);
bg.addColorStop(0,'#04060e');bg.addColorStop(0.2,'#080c18');bg.addColorStop(0.4,'#140a08');
bg.addColorStop(0.6,'#2a1006');bg.addColorStop(0.75,'#4a1a04');bg.addColorStop(0.9,'#803810');bg.addColorStop(1,'#c06020');
cx.fillStyle=bg;cx.fillRect(ML,MT,PW,PH);
cx.save();cx.beginPath();cx.rect(ML,MT,PW,PH);cx.clip();
var N=40,i;
cx.fillStyle='rgba(80,200,120,0.06)';cx.beginPath();
cx.moveTo(cX(0),tYf(0));cx.lineTo(cX(0.022),tYf(0));cx.lineTo(cX(0.022),tYf(A1));cx.lineTo(cX(0),tYf(A1));cx.fill();
cx.fillStyle='rgba(74,168,192,0.07)';cx.beginPath();
cx.moveTo(cX(0.022),tYf(0));cx.lineTo(cX(CE),tYf(0));cx.lineTo(cX(CE),tYf(A1));cx.lineTo(cX(0.022),tYf(A1));cx.fill();
cx.fillStyle='rgba(154,104,196,0.07)';cx.beginPath();
cx.moveTo(cX(CE),tYf(0));cx.lineTo(cX(MXC),tYf(0));cx.lineTo(cX(MXC),tYf(A1));cx.lineTo(cX(CE),tYf(A1));cx.fill();
var tg=cx.createLinearGradient(0,tYf(A3F),0,tYf(A1));
tg.addColorStop(0,'rgba(232,144,48,0.03)');tg.addColorStop(1,'rgba(232,144,48,0.12)');
cx.fillStyle=tg;cx.beginPath();cx.moveTo(cX(0),tYf(A3F));
for(i=0;i<=N;i++){var c=CE*i/N;cx.lineTo(cX(c),tYf(a3(c)))}cx.lineTo(cX(CE),tYf(A1));cx.lineTo(cX(0),tYf(A1));cx.fill();
cx.fillStyle='rgba(240,200,48,0.06)';cx.beginPath();cx.moveTo(cX(0),tYf(MLT));cx.lineTo(cX(CP),tYf(PER));
for(i=0;i<=N;i++){c=CE+(CA-CE)*i/N;cx.lineTo(cX(c),tYf(acm(c)))}
cx.lineTo(cX(CE),tYf(A1));for(i=N;i>=0;i--){c=CE*i/N;cx.lineTo(cX(c),tYf(a3(c)))}cx.fill();
cx.fillStyle='rgba(255,238,160,0.06)';cx.beginPath();cx.moveTo(cX(0),tYf(MXT));cx.lineTo(cX(MXC),tYf(MXT));
for(i=N;i>=0;i--){c=MXC*i/N;cx.lineTo(cX(c),tYf(lq(c)))}cx.fill();
cx.strokeStyle='#d0952a';cx.lineWidth=1.4;cx.shadowColor='#d0952a';cx.shadowBlur=3;
cx.beginPath();for(i=0;i<=60;i++){c=CE*i/60;cx[i?'lineTo':'moveTo'](cX(c),tYf(a3(c)))}cx.stroke();
cx.strokeStyle='#c04848';cx.shadowColor='#c04848';
cx.beginPath();for(i=0;i<=60;i++){c=CE+(CA-CE)*i/60;cx[i?'lineTo':'moveTo'](cX(c),tYf(acm(c)))}cx.stroke();
cx.shadowBlur=0;cx.strokeStyle='rgba(128,170,238,0.45)';cx.lineWidth=1;cx.setLineDash([5,3]);
cx.beginPath();cx.moveTo(cX(0),tYf(A1));cx.lineTo(cX(MXC),tYf(A1));cx.stroke();cx.setLineDash([]);
cx.strokeStyle='rgba(255,240,160,0.50)';cx.lineWidth=1.4;
cx.beginPath();for(i=0;i<=60;i++){c=MXC*i/60;cx[i?'lineTo':'moveTo'](cX(c),tYf(lq(c)))}cx.stroke();
cx.strokeStyle='rgba(80,227,194,0.30)';cx.lineWidth=1;cx.setLineDash([3,3]);
cx.beginPath();cx.moveTo(cX(D.C),tYf(0));cx.lineTo(cX(D.C),tYf(MXT));cx.stroke();cx.setLineDash([]);
cx.strokeStyle='rgba(255,107,53,0.22)';cx.lineWidth=1.5;
cx.beginPath();for(i=0;i<D.path.length;i++){var pt=D.path[i];cx[i?'lineTo':'moveTo'](cX(D.C),tYf(pt.T))}cx.stroke();
var T=F[ci].T,ox=cX(D.C),oy=tYf(T);
cx.fillStyle='rgba(255,107,53,0.12)';cx.beginPath();cx.arc(ox,oy,16,0,6.28);cx.fill();
cx.fillStyle='rgba(255,107,53,0.25)';cx.beginPath();cx.arc(ox,oy,9,0,6.28);cx.fill();
cx.fillStyle='#ff6b35';cx.shadowColor='#ff6b35';cx.shadowBlur=8;cx.beginPath();cx.arc(ox,oy,4.5,0,6.28);cx.fill();
cx.strokeStyle='rgba(255,255,255,0.80)';cx.lineWidth=1.5;cx.shadowBlur=0;cx.beginPath();cx.arc(ox,oy,4.5,0,6.28);cx.stroke();
cx.restore();
cx.strokeStyle=(D.theme&&D.theme.grid)||'rgba(160,185,220,0.10)';cx.lineWidth=0.5;
for(T=0;T<=MXT;T+=400){cx.beginPath();cx.moveTo(ML,tYf(T));cx.lineTo(ML+PW,tYf(T));cx.stroke()}
cx.fillStyle=(D.theme&&D.theme.text)||'rgba(160,185,220,0.62)';cx.font='9px monospace';cx.textAlign='right';
for(T=0;T<=MXT;T+=400)cx.fillText(T+'\u00b0',ML-4,tYf(T)+3);
cx.textAlign='center';cx.fillText('wt% C',ML+PW/2,H-4);
for(c=0;c<=MXC;c+=0.5)cx.fillText(c.toFixed(1),cX(c),H-MB+14);
cx.fillStyle='rgba(200,220,245,0.28)';cx.font='bold 9px sans-serif';cx.textAlign='center';
var lbs=[[0.38,280,'\u03b1+P'],[0.50,1020,'\u03b3'],[1.1,1400,'L']];
for(i=0;i<lbs.length;i++)cx.fillText(lbs[i][2],cX(lbs[i][0]),tYf(lbs[i][1]));
cx.strokeStyle='rgba(160,185,220,0.10)';cx.lineWidth=1;cx.strokeRect(ML,MT,PW,PH)}
function setFrame(idx){
ci=Math.max(0,Math.min(NF-1,idx));var f=F[ci];
stageEl.textContent=f.s;tvalEl.textContent=Math.round(f.T);
mimg.src=f.img;
var pct=ci/(NF-1)*100;sfill.style.width=pct+'%';sthumb.style.left=pct+'%';
ftime.textContent=(ci+1)+' / '+NF;
phbar.innerHTML='';phleg.innerHTML='';
for(var j=0;j<f.ph.length;j++){var p=f.ph[j];
var seg=document.createElement('div');seg.className='phseg';
seg.style.width=(p.f*100)+'%';seg.style.background=p.c;phbar.appendChild(seg);
var lg=document.createElement('div');lg.className='phl';
lg.innerHTML='<div class="phd" style="background:'+p.c+'"></div>'+p.n+' '+(p.f*100).toFixed(0)+'%';
phleg.appendChild(lg)}
drawPhase()}
function togglePlay(){
if(playing){playing=false;clearInterval(timer);pbtn.innerHTML='&#9654;'}
else{playing=true;pbtn.innerHTML='&#10074;&#10074;';
timer=setInterval(function(){if(ci>=NF-1){ci=0}else{ci++}setFrame(ci)},140)}}
pbtn.addEventListener('click',togglePlay);
var dragging=false;
function scrub(e){
var r=sbar.getBoundingClientRect();var x=Math.max(0,Math.min(1,(e.clientX-r.left)/r.width));
setFrame(Math.round(x*(NF-1)))}
sbar.addEventListener('mousedown',function(e){dragging=true;scrub(e)});
window.addEventListener('mousemove',function(e){if(dragging)scrub(e)});
window.addEventListener('mouseup',function(){dragging=false});
sbar.addEventListener('touchstart',function(e){e.preventDefault();var t=e.touches[0];
var r=sbar.getBoundingClientRect();var x=Math.max(0,Math.min(1,(t.clientX-r.left)/r.width));
setFrame(Math.round(x*(NF-1)))},{passive:false});
sbar.addEventListener('touchmove',function(e){e.preventDefault();var t=e.touches[0];
var r=sbar.getBoundingClientRect();var x=Math.max(0,Math.min(1,(t.clientX-r.left)/r.width));
setFrame(Math.round(x*(NF-1)))},{passive:false});
var imgs=[];var loaded=0;
for(var k=0;k<NF;k++){var im=new Image();im.src=F[k].img;im.onload=function(){loaded++};imgs.push(im)}
setFrame(0);
</script></body></html>"""



# ══════════════════════════════════════════════════════════════════════════════
#  HEADER  —  brand wordmark + theme toggle
# ══════════════════════════════════════════════════════════════════════════════
hdr_l, hdr_c, hdr_r = st.columns([1, 4, 1])
with hdr_r:
    if st.button(("☾ DARK" if not IS_DARK else "☀ LIGHT"),
                 key="miq_theme_toggle",
                 type="secondary",
                 help="Toggle between dark and light theme"):
        st.session_state.miq_theme = "light" if IS_DARK else "dark"
        st.rerun()
with hdr_c:
    st.markdown(
        f'<div style="text-align:center;padding:1.6rem 0 0.8rem;">'
        f'<div class="miq-hero-title" style="font-family:\'Rajdhani\',sans-serif;font-size:3.6rem;'
        f'font-weight:700;background:linear-gradient(135deg,{T["hero_a"]} 0%,{T["hero_b"]} 45%,{T["hero_c"]} 100%);'
        f'-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;'
        f'letter-spacing:0.08em;line-height:1;">MAST&nbsp;IQ</div>'
        f'<div class="miq-hero-sub" style="color:var(--fg-secondary);font-size:0.84rem;'
        f'letter-spacing:0.28em;text-transform:uppercase;margin-top:8px;font-weight:500;">'
        f'Alloy Intelligence Platform &nbsp;&middot;&nbsp; mast lab</div>'
        f'<div style="width:140px;height:2px;'
        f'background:linear-gradient(90deg,transparent,{T["accent"]},{T["accent_cool"]},transparent);'
        f'margin:14px auto 0;border-radius:2px;"></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — INPUTS
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        f'<div style="text-align:center;padding:4px 0 12px;">'
        f'<span style="font-family:\'Rajdhani\',sans-serif;font-size:1.20rem;'
        f'font-weight:700;background:linear-gradient(90deg,{T["accent"]},{T["accent_cool"]});'
        f'-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;'
        f'letter-spacing:0.18em;">INPUT&nbsp;PARAMETERS</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("**PROCESS TYPE**")
    PROCESS_MAP = {
        "Quench & Temper": "Quench_Temper",
        "Normalizing":     "Normalizing",
        "Full Annealing":  "Annealing",
        "Stress Relief":   "Stress_Relief",
    }
    process_label = st.selectbox(
        "Heat Treatment Process", list(PROCESS_MAP.keys()),
        label_visibility="collapsed",
    )
    process_key = PROCESS_MAP[process_label]

    st.markdown("---")
    st.markdown("**CHEMICAL COMPOSITION (wt%)**")

    col1, col2 = st.columns(2)
    with col1:
        C  = st.number_input("C",  value=0.40, min_value=0.05, max_value=0.60, step=0.01, format="%.3f")
        Mn = st.number_input("Mn", value=0.85, min_value=0.10, max_value=2.00, step=0.05, format="%.3f")
        Ni = st.number_input("Ni", value=0.20, min_value=0.00, max_value=4.00, step=0.10, format="%.3f")
        Cu = st.number_input("Cu", value=0.10, min_value=0.00, max_value=0.50, step=0.05, format="%.3f")
        P  = st.number_input("P",  value=0.010, min_value=0.001, max_value=0.040, step=0.001, format="%.3f")
    with col2:
        Si = st.number_input("Si", value=0.25, min_value=0.05, max_value=1.00, step=0.05, format="%.3f")
        Cr = st.number_input("Cr", value=1.05, min_value=0.00, max_value=2.00, step=0.10, format="%.3f")
        Mo = st.number_input("Mo", value=0.20, min_value=0.00, max_value=0.70, step=0.05, format="%.3f")
        S  = st.number_input("S",  value=0.010, min_value=0.001, max_value=0.040, step=0.001, format="%.3f")

    CE  = calc_carbon_equiv(C, Mn, Si, Ni, Cr, Mo, Cu)
    A3v = calc_a3(C, Mn, Si, Ni, Cr, Mo)
    Fe  = 100.0 - (C + Si + Mn + P + S + Ni + Cr + Cu + Mo)
    st.markdown(
        f'<div class="miq-card" style="padding:10px 13px;margin:6px 0;border-left:3px solid var(--accent);">'
        f'<div style="display:grid;grid-template-columns:auto 1fr;gap:5px 10px;font-size:0.77rem;">'
        f'<span style="color:var(--fg-secondary);">Fe (balance)</span>'
        f'<span style="font-family:\'JetBrains Mono\',monospace;color:var(--accent);font-weight:600;text-align:right;">{Fe:.2f} wt%</span>'
        f'<span style="color:var(--fg-secondary);">CE (IIW)</span>'
        f'<span style="font-family:\'JetBrains Mono\',monospace;color:var(--accent-cool);font-weight:600;text-align:right;">{CE:.4f}</span>'
        f'<span style="color:var(--fg-secondary);">A3 temp</span>'
        f'<span style="font-family:\'JetBrains Mono\',monospace;color:var(--accent-warn);font-weight:600;text-align:right;">{A3v:.0f}\u202f\u00b0C</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(f"**{process_label.upper()} PARAMETERS**")

    if process_key == "Quench_Temper":
        ht_temp   = st.slider("Austenitize Temp (C)", 780, 980, int(max(A3v+40, 830)), 5)
        soak_time = st.slider("Soaking Time (min)", 15, 120, 45, 5)
        cool_medium = st.selectbox("Quench Medium", ["Oil", "Water", "Polymer", "Salt Bath"])
        t_temp    = st.slider("Tempering Temp (C)", 150, 700, 550, 10)
        t_time    = st.slider("Tempering Time (min)", 30, 240, 120, 10)
        if ht_temp >= A3v:
            st.success(f"OK {ht_temp}C > A3 ({A3v:.0f}C) - fully austenitized")
        else:
            st.warning(f"WARN {ht_temp}C < A3 ({A3v:.0f}C) - incomplete austenitizing")
        if t_temp >= 723:
            st.error("Tempering temp >= A1 - risk of re-austenitizing!")
        HJ = calc_hollomon_jaffe(t_temp, t_time)
        st.markdown(
            f'<div class="miq-card" style="padding:8px 13px;margin-top:6px;border-left:3px solid var(--accent-cool);'
            f'display:flex;justify-content:space-between;align-items:center;">'
            f'<span style="font-size:0.75rem;color:var(--fg-secondary);">Hollomon-Jaffe&nbsp;H</span>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;color:var(--accent-cool);font-weight:600;">{HJ:,.0f}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    elif process_key == "Normalizing":
        ht_temp   = st.slider("Normalizing Temp (C)", 800, 980, int(max(A3v+30, 820)), 5)
        soak_time = st.slider("Soaking Time (min)", 10, 90, 30, 5)
        cool_medium = "Air"
        t_temp, t_time = 0.0, 0.0
        st.info("Cooling medium: Air  (standard normalizing)")
        if ht_temp >= A3v:
            st.success(f"OK {ht_temp}C > A3 ({A3v:.0f}C) - fully austenitized")
        else:
            st.warning("Below A3 - incomplete normalizing")

    elif process_key == "Annealing":
        ht_temp   = st.slider("Annealing Temp (C)", 800, 980, int(max(A3v+30, 820)), 5)
        soak_time = st.slider("Soaking Time (min)", 30, 150, 60, 10)
        cool_medium = "Furnace"
        t_temp, t_time = 0.0, 0.0
        st.info("Cooling medium: Furnace  (controlled slow cool)")
        if ht_temp >= A3v:
            st.success(f"OK {ht_temp}C > A3 ({A3v:.0f}C) - fully austenitized")
        else:
            st.warning("Below A3 - incomplete annealing")

    else:  # Stress Relief
        ht_temp   = st.slider("SR Temp (C)", 150, 700, 450, 10)
        soak_time = st.slider("Soaking Time (min)", 60, 300, 120, 15)
        cool_medium = "Air"
        t_temp, t_time = 0.0, 0.0
        st.info("Cooling medium: Air")
        if ht_temp >= 723:
            st.error("SR temp >= A1 = 723C - not stress relief!")
        else:
            st.success(f"OK {ht_temp}C < A1 (723C) - no phase change")

    st.markdown("---")
    predict_btn = st.button("PREDICT PROPERTIES", type="primary", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE  —  predictions are ONLY computed on button click (snapshot)
# ══════════════════════════════════════════════════════════════════════════════
if "show_results" not in st.session_state:
    st.session_state.show_results = False
if "snap" not in st.session_state:
    st.session_state.snap = None

if predict_btn:
    _feat  = build_feature_vector(
        process_key, C, Si, Mn, P, S, Ni, Cr, Cu, Mo,
        float(ht_temp), float(soak_time), cool_medium,
        float(t_temp), float(t_time),
    )
    _preds = predict(_feat)
    st.session_state.show_results = True
    st.session_state.snap = {
        "feat": _feat, "preds": _preds,
        "process_key": process_key, "process_label": process_label,
        "C": C, "Si": Si, "Mn": Mn, "P": P, "S": S,
        "Ni": Ni, "Cr": Cr, "Cu": Cu, "Mo": Mo,
        "ht_temp": float(ht_temp), "soak_time": float(soak_time),
        "cool_medium": cool_medium,
        "t_temp": float(t_temp), "t_time": float(t_time),
        "CE": CE, "A3v": A3v,
    }
    st.session_state.pop("sim_ready",  None)
    st.session_state.pop("sim_params", None)


# ══════════════════════════════════════════════════════════════════════════════
#  WELCOME SCREEN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="miq-hint">'
    '<span>☰</span>'
    '<span>Tap the menu icon (top-left) to open <b>Input Parameters</b>.</span>'
    '</div>',
    unsafe_allow_html=True,
)

if not st.session_state.show_results:
    col_a, col_b, col_c, col_d = st.columns(4)
    for col, proc, key, color, icon, desc in [
        (col_a, "Quench & Temper", "Quench_Temper", "#4c72b0", "🔥",
         "Austenitize then rapid quench then temper. Highest strength and hardness."),
        (col_b, "Normalizing",     "Normalizing",   "#50c878", "🌬",
         "Austenitize then air cool. Refines grain size, uniform properties."),
        (col_c, "Full Annealing",  "Annealing",     "#f0a030", "🔆",
         "Austenitize then furnace cool. Maximum softness, high ductility."),
        (col_d, "Stress Relief",   "Stress_Relief", "#e05060", "🛡",
         "Sub-A1 heat then air cool. Relieves residual stress, no phase change."),
    ]:
        with col:
            st.markdown(html_process_card(proc, color, icon, desc),
                        unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        html_section_header("How to use MAST IQ",
                             "Select process  ·  Enter composition  ·  Set parameters  ·  Click PREDICT",
                             "📖"),
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    for col, num, ttl, body in [
        (c1, "01", "SELECT PROCESS",
         "Choose a heat treatment method from the sidebar. The input form adapts automatically "
         "to show relevant parameters for that process."),
        (c2, "02", "ENTER COMPOSITION",
         "Input alloy chemistry in wt%. A3 temperature and carbon equivalent are computed "
         "live as metallurgical guidance."),
        (c3, "03", "PREDICT AND EXPLORE",
         "Click PREDICT PROPERTIES to get Tensile, Yield, Hardness, Elongation and Fatigue "
         "predictions with interactive charts and a live phase and microstructure animation."),
    ]:
        with col:
            st.markdown(
                f'<div class="miq-card" style="padding:18px 18px;height:100%;">'
                f'<div style="font-family:\'Rajdhani\',sans-serif;font-size:1.05rem;'
                f'color:var(--accent);font-weight:700;letter-spacing:0.10em;margin-bottom:8px;">'
                f'{num} &mdash; {ttl}</div>'
                f'<div style="font-size:0.80rem;color:var(--fg-secondary);line-height:1.55;">{body}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
#  RESULTS SCREEN  (driven entirely by the frozen snapshot)
# ══════════════════════════════════════════════════════════════════════════════
else:
    snap = st.session_state.snap
    if snap is None:
        st.warning("No prediction found. Click **PREDICT PROPERTIES** in the sidebar.")
        st.stop()

    # Unpack snapshot
    feat     = snap["feat"]
    preds    = snap["preds"]
    pkey     = snap["process_key"]
    plabel   = snap["process_label"]
    C_s      = snap["C"];  Mn_s = snap["Mn"]; Cr_s = snap["Cr"]; Mo_s = snap["Mo"]
    ht_s     = snap["ht_temp"];  sk_s  = snap["soak_time"]
    cool_s   = snap["cool_medium"]
    tt_s     = snap["t_temp"];   ttime_s = snap["t_time"]
    CE_s     = snap["CE"];       A3v_s   = snap["A3v"]

    # OOD warnings (collapsed expander — not an intrusive modal)
    ood = check_ood(feat)
    if ood:
        with st.expander(f"Input range notice ({len(ood)} feature(s) outside training data)",
                         expanded=False):
            for msg in ood:
                st.warning(msg)
            st.caption("Extrapolation warning: predictions may be less reliable. "
                       "Validate against certified testing for critical applications.")

    # Process banner
    PROC_COLORS = {"Quench_Temper":"#4c72b0","Normalizing":"#50c878",
                   "Annealing":"#f0a030","Stress_Relief":"#e05060"}
    pc          = PROC_COLORS.get(pkey, "#4c72b0")
    grade, gcol = steel_grade(preds["Tensile_MPa"])
    phase_at_ht = get_phase(C_s, ht_s)
    phase_color = PHASE_COLORS.get(phase_at_ht, "#aaa")
    temper_str  = (f"  to  {int(tt_s)}C temper") if tt_s > 0 else ""
    
    yt_ratio = preds["Yield_MPa"] / preds["Tensile_MPa"] if preds.get("Tensile_MPa") else 0
    yt_color = "#e05060" if yt_ratio > 0.90 else "#50c878"
    yt_warn = "Warning" if yt_ratio > 0.90 else "Good"

    st.markdown(
        f'<div class="miq-card" style="background:linear-gradient(135deg,{pc}1a,var(--bg-surface));'
        f'border-left:4px solid {pc};margin-bottom:18px;'
        f'display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;">'
        f'<div>'
        f'<span style="font-family:\'Rajdhani\',sans-serif;font-size:1.30rem;font-weight:700;color:{pc};letter-spacing:0.04em;">'
        f'{plabel}</span>'
        f'<span style="font-size:0.78rem;color:var(--fg-secondary);margin-left:14px;">'
        f'{ht_s:.0f}\u00b0C &nbsp;&middot;&nbsp; {sk_s:.0f}\u202fmin &nbsp;&middot;&nbsp; {cool_s}{temper_str}</span>'
        f'</div>'
        f'<div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;">'
        f'<div style="display:flex;flex-direction:column;align-items:flex-end;margin-right:6px;">'
        f'<span style="font-size:0.65rem;letter-spacing:0.10em;color:var(--fg-muted);text-transform:uppercase;">Y/T Ratio</span>'
        f'<span style="font-family:\'Rajdhani\',sans-serif;font-size:1.0rem;font-weight:700;color:{yt_color};">{yt_ratio:.2f} '
        f'<span style="font-size:0.72rem;font-family:\'Inter\',sans-serif;font-weight:500;">({yt_warn})</span></span>'
        f'</div>'
        f'<span style="background:{gcol}22;border:1px solid {gcol}50;color:{gcol};'
        f'border-radius:20px;padding:3px 11px;font-size:0.73rem;font-weight:600;">{grade}</span>'
        f'<span style="background:{phase_color}22;border:1px solid {phase_color}50;color:{phase_color};'
        f'border-radius:20px;padding:3px 11px;font-size:0.73rem;font-weight:600;">{phase_at_ht}</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Metric cards — CSS grid auto-wraps on mobile
    cards_html = ""
    for tgt in TARGETS:
        label, unit, icon, color = TARGET_LABELS[tgt]
        val  = preds[tgt]
        disp = f"{val:.1f}" if tgt == "Elongation_pct" else f"{val:.0f}"
        cards_html += html_metric_card(label, disp, unit, color, icon)
    st.markdown(f'<div class="miq-cards">{cards_html}</div>', unsafe_allow_html=True)

    # Dataset disclaimer
    st.markdown(
        '<div class="miq-card" style="padding:10px 16px;margin-bottom:14px;font-size:0.78rem;'
        'color:var(--fg-secondary);border-left:3px solid var(--accent-cool);">'
        '<b style="color:var(--fg-primary);">Model disclaimer.</b> '
        'Predictions are ML estimates from curated heat-treatment datasets. '
        'For safety-critical or industrial certification, validate against '
        'standardised material testing (ASTM / ISO). Accuracy may vary outside training range.'
        '</div>',
        unsafe_allow_html=True,
    )

    report_data = {
        "Process": plabel,
        "Composition_wt_pct": {"C": C_s, "Mn": Mn_s, "Cr": Cr_s, "Mo": Mo_s},
        "Parameters": {"Austenitize_C": ht_s, "Soak_min": sk_s, "Quench": cool_s, "Temper_C": tt_s, "Temper_min": ttime_s},
        "Predictions": {k: float(v) for k, v in preds.items()},
        "Metallurgy": {"Y_T_Ratio": round(yt_ratio, 3), "Phase_at_HT": phase_at_ht, "Classification": grade}
    }
    col1, col2, col3 = st.columns([2,1,1])
    with col3:
        st.download_button(
            label="📥 Download JSON Report",
            data=json_lib.dumps(report_data, indent=2),
            file_name=f"MAST_IQ_{plabel.replace(' ', '_')}_Report.json",
            mime="application/json",
            use_container_width=True,
        )

    # Tabs
    tab1, tab2, tab3, tab_cct, tab4, tab5 = st.tabs([
        "⟐ Analysis",
        "⬢ Phase Diagram",
        "⎍ T-t Profile",
        "◎ CCT Diagram",
        "≡ Input Summary",
        "▶ Live Simulation",
    ])

    # Tab 1: Performance Analysis (radar + metallurgical context — no duplicate gauges/tables)
    with tab1:
        c_r, c_s = st.columns([1.15, 1])
        with c_r:
            st.markdown(html_section_header("Performance Radar",
                                             "Normalised 0-100% relative to property range", "🕸"),
                        unsafe_allow_html=True)
            st.plotly_chart(build_radar(preds), width='stretch')
        with c_s:
            st.markdown(html_section_header("Metallurgical Context",
                                             "Derived physical-metallurgy quantities", "⚗️"),
                        unsafe_allow_html=True)
            meta_rows = [
                ("Carbon Equiv (IIW)",  f"{CE_s:.4f}",          "wt%"),
                ("A3 Temperature",      f"{A3v_s:.0f}",         "°C"),
                ("Δ T above A3",        f"{ht_s - A3v_s:+.0f}", "°C"),
                ("Phase at HT temp",    phase_at_ht,            ""),
            ]
            if tt_s > 0:
                HJ = calc_hollomon_jaffe(tt_s, ttime_s)
                meta_rows.append(("Hollomon-Jaffe H", f"{HJ:,.0f}", ""))
            st.dataframe(
                pd.DataFrame(meta_rows, columns=["Quantity","Value","Unit"]).set_index("Quantity"),
                width='stretch',
            )
            st.caption("Property values shown in cards above — this tab focuses on how the "
                       "predicted profile compares to the dataset average and the underlying "
                       "metallurgical state.")

    # Tab 2: Immersive Phase Diagram
    with tab2:
        phase_html = build_immersive_phase_html(
            C_s, ht_s, tt_s, pkey, cool_s,
            Mn=Mn_s, Cr=Cr_s, Mo=Mo_s,
        )
        components.html(phase_html, height=720, scrolling=False)
        st.caption(
            "Interactive phase visualisation \u2014 hover over any region to explore its physical behaviour. "
            "Animated particles represent the energy and structure of each phase. "
            "The operating point marks your current austenitising condition."
        )
        with st.expander("\U0001f4ca Classic Phase Diagram (Plotly)", expanded=False):
            st.plotly_chart(
                build_phase_diagram(C_s, ht_s, tt_s, pkey, cool_s,
                                     Mn=Mn_s, Cr=Cr_s, Mo=Mo_s),
                width='stretch',
            )

    # Tab 3: T-t Profile
    with tab3:
        st.plotly_chart(
            build_tt_profile(pkey, ht_s, tt_s, sk_s, ttime_s, C=C_s, Mn=Mn_s, Cr=Cr_s, Mo=Mo_s),
            width='stretch',
        )
        st.caption("Schematic temperature-time profile (times scaled for illustration). "
                   "A1 and A3 reference lines shown where applicable.")

    # Tab CCT: Continuous Cooling Transformation Diagram
    with tab_cct:
        st.markdown(html_section_header(
            "CCT Diagram",
            "Continuous Cooling Transformation curves for the current composition",
            "◎"),
            unsafe_allow_html=True)
        st.plotly_chart(
            build_cct_diagram(
                C=C_s, Mn=Mn_s, Si=snap.get("Si", 0.25),
                Ni=snap.get("Ni", 0.20), Cr=Cr_s, Mo=Mo_s,
                process_key=pkey, cool_medium=cool_s,
                ht_temp=ht_s,
            ),
            width='stretch',
        )
        st.caption(
            "Approximate CCT diagram based on Kirkaldy-type empirical model. "
            "C-curves show ferrite, pearlite, and bainite transformation start times. "
            "Cooling curves for all media are overlaid; the selected medium is highlighted. "
            "Ms and Bs lines are composition-specific. "
            "For precise CCT data, use JMatPro or Thermo-Calc."
        )

    # Tab 4: Input Summary
    with tab4:
        st.markdown(html_section_header("Full Input Vector Sent to Model", "", "🔬"),
                    unsafe_allow_html=True)
        inp_df = pd.DataFrame([feat]).T.rename(columns={0: "Value"})
        inp_df["Value"] = inp_df["Value"].apply(lambda v: f"{float(v):.4f}")
        st.dataframe(inp_df, width='stretch')

    # Tab 5: Combined Simulation
    with tab5:
        st.markdown(
            html_section_header(
                "Phase Diagram & Microstructure Simulation",
                "Interactive animation: Fe-C operating path + EBSD-style grain evolution.",
                "\U0001f3ac",
            ),
            unsafe_allow_html=True,
        )

        run_sim = st.button("Run Combined Simulation", type="primary")

        if run_sim:
            st.session_state.sim_ready  = True
            st.session_state.sim_params = (
                pkey, float(C_s), float(ht_s), float(sk_s),
                cool_s, float(tt_s), float(ttime_s),
            )

        if st.session_state.get("sim_ready"):
            sim_p = st.session_state.sim_params
            try:
                with st.spinner("Building immersive simulation \u2014 this may take up to 20 s..."):
                    sim_html = build_immersive_simulation_html(*sim_p, theme_key=st.session_state.miq_theme)
                components.html(sim_html, height=800, scrolling=False)
            except Exception as e:
                import traceback
                st.error(f"Simulation failed for process **{sim_p[0]}** \u2014 {type(e).__name__}: {e}")
                with st.expander("Debug traceback", expanded=False):
                    st.code(traceback.format_exc())

            _proc, _C, _ht, _soak, _cool, _tt, _ttime = sim_p
            Ms_v = max(80.0, 539 - 423*_C - 30.4*0.85 - 12.1*1.05 - 7.5*0.2)
            K_gg = 0.9 * np.exp(-20000 / (_ht + 273.15)) * (_soak / 60.0)
            n_s  = max(16, int(100 / (1.0 + 7.0 * K_gg)))
            with st.expander("Simulation physics parameters", expanded=False):
                phys_rows = [
                    ("Martensite start Ms",        f"{Ms_v:.0f}",                  "\u00b0C"),
                    ("Grain growth factor K_gg",   f"{K_gg:.4f}",                  ""),
                    ("Grain coarsening (approx.)", f"{round((1-n_s/100)*100,1)}",  "%"),
                    ("JMAK exponent",               "n=2.0 (Q+T)  n=1.8 (Ann.)",  ""),
                    ("Cooling speed class",         _cool,                          ""),
                    ("Voronoi grains (soaked)",     str(n_s),                       ""),
                ]
                st.dataframe(
                    pd.DataFrame(phys_rows, columns=["Parameter","Value","Unit"]).set_index("Parameter"),
                    width='stretch',
                )
        else:
            st.info("Click Run Combined Simulation to generate the interactive "
                    "Fe-C path and microstructure animation for the current prediction.")

