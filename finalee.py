"""
MAST IQ — Alloy Intelligence Platform
=======================================
By MAST  ·  Materials Advanced Science & Technology
AI-Powered Steel Heat Treatment Mechanical Property Predictor

Targets  : Tensile · Yield · Hardness · Elongation · Fatigue
Processes: Quench & Temper · Normalizing · Full Annealing · Stress Relief
"""

import os, json as json_lib, base64
import streamlit as st
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
#  DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
# Theme variant from sidebar toggle (read from session_state set by the toggle widget)
_THEME_CRYO = st.session_state.get("miq_theme_cryo", False)
_A_HOT  = "#6fa8ff" if _THEME_CRYO else "#ff7b2e"
_A_WARM = "#89c8ff" if _THEME_CRYO else "#ffb840"
_A_COOL = "#3db8ff"
_A_PLSM = "#8a6cff"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Orbitron:wght@500;700&display=swap');

:root {{
    --bg-0: #020510;
    --bg-1: #060c1c;
    --bg-2: #0a1428;
    --fg-0: #e8f0ff;
    --fg-1: #a8c0e5;
    --fg-2: #6a88b2;
    --accent-hot: {_A_HOT};
    --accent-warm: {_A_WARM};
    --accent-cool: {_A_COOL};
    --accent-plasma: {_A_PLSM};
    --border-dim: rgba(100,160,240,0.14);
    --border-mid: rgba(100,160,240,0.24);
    --card-bg: rgba(8,14,28,0.72);
}}

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif !important; color: var(--fg-1) !important; }}

/* Hide Streamlit chrome but keep sidebar toggle accessible (critical for mobile) */
#MainMenu, footer {{ visibility: hidden !important; }}
[data-testid="stDecoration"] {{ display: none !important; }}
[data-testid="stToolbar"] {{ visibility: hidden !important; }}
header[data-testid="stHeader"] {{ background: transparent !important; height: auto !important; }}
[data-testid="collapsedControl"] {{
    visibility: visible !important;
    display: flex !important;
    background: rgba(8,14,28,0.90) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: 10px !important;
    backdrop-filter: blur(10px) !important;
    box-shadow: 0 0 18px rgba(61,184,255,0.22) !important;
}}

/* ── Dynamic molten + plasma background ── */
.stApp {{
    background:
        radial-gradient(ellipse 70% 45% at 16% 6%,  rgba(255,123,46,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 65% 45% at 86% 94%, rgba(61,184,255,0.08) 0%, transparent 62%),
        radial-gradient(ellipse 100% 80% at 50% 50%, #060c1c 0%, #020510 68%, #000208 100%);
    min-height: 100vh;
    position: relative;
}}
.stApp::before {{
    content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background: conic-gradient(from 0deg at 50% 50%,
        transparent 0deg,
        rgba(61,184,255,0.022) 60deg,
        transparent 120deg,
        rgba(255,123,46,0.020) 240deg,
        transparent 320deg);
    animation: miq-sweep 38s linear infinite;
}}
.stApp::after {{
    content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image:
        radial-gradient(circle at 22% 28%, rgba(61,184,255,0.045) 0%, transparent 28%),
        radial-gradient(circle at 78% 72%, rgba(255,123,46,0.038) 0%, transparent 28%);
    animation: miq-pulse 9s ease-in-out infinite alternate;
}}
@keyframes miq-sweep {{ from{{transform:rotate(0)}} to{{transform:rotate(360deg)}} }}
@keyframes miq-pulse {{ from{{opacity:0.55}} to{{opacity:1}} }}
@keyframes miq-shimmer {{
    0%  {{ background-position: -200% 0; }}
    100% {{ background-position: 200% 0; }}
}}

[data-testid="stAppViewContainer"] > .main {{ position: relative; z-index: 1; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg,
        rgba(6,12,26,0.97) 0%,
        rgba(10,16,32,0.94) 45%,
        rgba(4,8,18,0.98) 100%) !important;
    border-right: 1px solid var(--border-dim) !important;
    backdrop-filter: blur(14px) !important;
}}
[data-testid="stSidebar"]::before {{
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-hot) 22%,
                var(--accent-cool) 78%, transparent);
    opacity: 0.8; z-index: 10;
}}
[data-testid="stSidebar"] .block-container {{ padding: 1.0rem 0.95rem !important; }}
[data-testid="stSidebar"] label {{
    color: var(--fg-2) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
}}

/* ── Number input ── */
.stNumberInput input {{
    background: rgba(4,10,24,0.92) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: 8px !important;
    color: var(--fg-0) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.86rem !important;
    transition: all 0.22s ease !important;
}}
.stNumberInput input:focus {{
    border-color: var(--accent-cool) !important;
    box-shadow: 0 0 0 3px rgba(61,184,255,0.13), 0 0 24px rgba(61,184,255,0.22) !important;
}}
.stNumberInput button {{
    background: rgba(4,10,24,0.92) !important;
    border: 1px solid var(--border-dim) !important;
    color: var(--accent-cool) !important;
}}

/* ── Sliders ── */
[data-testid="stSlider"] > div > div > div > div {{
    background: linear-gradient(90deg, var(--accent-hot) 0%, var(--accent-cool) 100%) !important;
    height: 5px !important;
}}
[data-testid="stSlider"] [role="slider"] {{
    background: var(--fg-0) !important;
    border: 2px solid var(--accent-cool) !important;
    box-shadow: 0 0 14px rgba(61,184,255,0.60), 0 0 6px rgba(255,123,46,0.35) !important;
    width: 18px !important; height: 18px !important;
}}
[data-testid="stSlider"] p {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    color: var(--accent-cool) !important;
}}

/* ── Select ── */
.stSelectbox > div > div {{
    background: rgba(4,10,24,0.92) !important;
    border: 1px solid var(--border-mid) !important;
    border-radius: 8px !important;
    color: var(--fg-0) !important;
}}

/* ── Toggle (theme switcher) ── */
[data-testid="stToggle"] label p {{
    color: var(--fg-1) !important;
    font-size: 0.74rem !important;
    letter-spacing: 0.10em !important;
    font-weight: 600 !important;
}}
[data-testid="stToggle"] [data-baseweb="toggle"] > div:first-child {{
    background: linear-gradient(90deg, var(--accent-hot), var(--accent-cool)) !important;
}}

/* ── Primary button (with shimmer sweep on hover) ── */
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, #002a60 0%, #004db0 42%, #005bcc 72%, #003680 100%) !important;
    border: 1px solid rgba(61,184,255,0.55) !important;
    color: var(--fg-0) !important;
    border-radius: 11px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    padding: 0.78rem 1.4rem !important;
    box-shadow: 0 6px 30px rgba(0,90,204,0.45),
                inset 0 1px 0 rgba(255,255,255,0.14) !important;
    transition: all 0.28s cubic-bezier(.2,.8,.4,1) !important;
    width: 100% !important;
    position: relative !important;
    overflow: hidden !important;
}}
.stButton > button[kind="primary"]::before {{
    content: ""; position: absolute; top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.22), transparent);
    transition: left 0.7s ease;
}}
.stButton > button[kind="primary"]:hover::before {{ left: 100%; }}
.stButton > button[kind="primary"]:hover {{
    background: linear-gradient(135deg, #003680 0%, #006ad0 42%, #0080e8 72%, #0052b0 100%) !important;
    box-shadow: 0 8px 38px rgba(0,140,255,0.58),
                inset 0 1px 0 rgba(255,255,255,0.22),
                0 0 0 1px rgba(61,184,255,0.55) !important;
    transform: translateY(-2px) !important;
}}
.stButton > button[kind="primary"]:active {{ transform: translateY(-1px) scale(0.99) !important; }}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: rgba(4,10,24,0.90) !important;
    border: 1px solid var(--border-dim) !important;
    border-radius: 14px !important;
    padding: 5px !important; gap: 3px !important;
    overflow-x: auto !important; flex-wrap: nowrap !important;
    backdrop-filter: blur(10px) !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.32) !important;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 9px !important;
    color: var(--fg-2) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.44rem 1.10rem !important;
    white-space: nowrap !important;
    transition: all 0.22s !important;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: var(--fg-1) !important;
    background: rgba(61,184,255,0.08) !important;
}}
.stTabs [data-baseweb="tab"][aria-selected="true"] {{
    background: linear-gradient(135deg, rgba(255,123,46,0.18), rgba(61,184,255,0.22)) !important;
    color: var(--fg-0) !important;
    box-shadow: 0 0 22px rgba(61,184,255,0.24),
                inset 0 1px 0 rgba(255,255,255,0.10) !important;
}}

/* ── Expander ── */
[data-testid="stExpander"] {{
    background: var(--card-bg) !important;
    border: 1px solid var(--border-dim) !important;
    border-radius: 11px !important;
    backdrop-filter: blur(8px) !important;
}}
[data-testid="stExpander"] summary {{ color: var(--fg-1) !important; font-size: 0.82rem !important; }}

/* ── DataFrame ── */
.stDataFrame {{
    border-radius: 10px !important;
    overflow: hidden !important;
    border: 1px solid var(--border-dim) !important;
}}

hr {{ border: none !important; border-top: 1px solid var(--border-dim) !important; margin: 14px 0 !important; }}

.block-container {{
    padding: 1.5rem 2.0rem 1rem !important;
    max-width: 1440px !important;
    position: relative;
    z-index: 1;
}}

/* ── Metric cards grid ── */
.miq-cards {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(168px, 1fr));
    gap: 14px;
    margin: 0 0 18px;
}}

.miq-mobile-hint {{ display: none; }}

/* ── Mobile breakpoints ── */
@media (max-width: 1024px) {{
    .block-container {{ padding: 1.0rem 1.2rem 0.8rem !important; }}
}}
@media (max-width: 768px) {{
    .block-container {{ padding: 0.75rem 0.85rem 0.6rem !important; }}
    .stTabs [data-baseweb="tab"] {{
        font-size: 0.68rem !important;
        padding: 0.32rem 0.62rem !important;
        letter-spacing: 0.04em !important;
    }}
    .miq-hero-title {{ font-size: 2.1rem !important; letter-spacing: 0.04em !important; }}
    .miq-hero-sub {{ font-size: 0.66rem !important; letter-spacing: 0.14em !important; }}
    .miq-cards {{ grid-template-columns: repeat(2, 1fr) !important; gap: 10px !important; }}
    [data-testid="collapsedControl"] {{
        top: 0.6rem !important; left: 0.6rem !important;
        width: 44px !important; height: 44px !important;
    }}
    .miq-mobile-hint {{
        display: flex !important;
        align-items: center;
        gap: 10px;
        background: linear-gradient(135deg, rgba(255,123,46,0.14), rgba(61,184,255,0.14));
        border: 1px solid var(--border-mid);
        border-radius: 10px;
        padding: 10px 14px;
        margin: 0 0 14px;
        font-size: 0.76rem;
        color: var(--fg-1);
    }}
}}
@media (max-width: 480px) {{
    .miq-hero-title {{ font-size: 1.6rem !important; }}
    .miq-hero-sub {{ font-size: 0.56rem !important; letter-spacing: 0.10em !important; }}
    .miq-cards {{ grid-template-columns: repeat(2, 1fr) !important; gap: 8px !important; }}
    .block-container {{ padding: 0.45rem !important; }}
}}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HTML HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def html_metric_card(title, value, unit, color, icon, subtitle=""):
    sub = (f'<div style="font-size:0.67rem;color:rgba(150,190,240,0.30);margin-top:4px;">'
           f'{subtitle}</div>') if subtitle else ""
    return (
        f'<div style="background:linear-gradient(145deg,rgba(6,12,34,0.97),rgba(4,7,18,0.99));'
        f'border:1px solid {color}44;border-top:3px solid {color};border-radius:14px;'
        f'padding:16px 12px 14px;text-align:center;'
        f'box-shadow:0 6px 26px rgba(0,0,0,0.55),0 0 22px {color}12;'
        f'position:relative;overflow:hidden;">'
        f'<div style="position:absolute;top:-26px;right:-14px;font-size:5rem;opacity:0.05;">{icon}</div>'
        f'<div style="font-size:1.6rem;margin-bottom:3px;line-height:1;">{icon}</div>'
        f'<div style="font-size:2.15rem;font-weight:700;color:{color};'
        f'font-family:\'Rajdhani\',sans-serif;line-height:1.05;letter-spacing:-0.01em;">{value}</div>'
        f'<div style="font-size:0.64rem;color:rgba(150,190,240,0.40);text-transform:uppercase;'
        f'letter-spacing:0.16em;margin-top:6px;font-weight:600;">{title}</div>'
        f'<div style="font-size:0.72rem;color:{color}88;margin-top:2px;">{unit}</div>'
        f'{sub}</div>'
    )


def html_section_header(title, subtitle="", icon=""):
    sub = (f'<p style="color:rgba(150,190,240,0.48);font-size:0.80rem;margin:0 0 0 2.1rem;">'
           f'{subtitle}</p>') if subtitle else ""
    return (
        f'<div style="margin:0 0 14px 0;">'
        f'<div style="display:flex;align-items:center;gap:9px;margin-bottom:3px;">'
        f'<span style="font-size:1.2rem;">{icon}</span>'
        f'<span style="font-family:\'Rajdhani\',sans-serif;font-size:1.30rem;'
        f'font-weight:700;color:#ddf0ff;letter-spacing:0.03em;">{title}</span>'
        f'</div>{sub}</div>'
    )


def html_badge(text, color):
    return (f'<span style="display:inline-block;padding:3px 10px;border-radius:20px;'
            f'background:{color}28;border:1px solid {color}55;color:{color};'
            f'font-size:0.70rem;font-weight:600;letter-spacing:0.07em;'
            f'text-transform:uppercase;">{text}</span>')


def html_process_card(proc, color, icon, desc):
    return (
        f'<div style="background:rgba(6,12,34,0.75);border:1px solid {color}40;'
        f'border-left:3px solid {color};border-radius:10px;padding:12px 14px;margin-bottom:6px;">'
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
        f'<span style="font-size:1.05rem;">{icon}</span>'
        f'<span style="font-family:\'Rajdhani\',sans-serif;font-size:1.02rem;'
        f'font-weight:700;color:{color};">{proc}</span></div>'
        f'<div style="font-size:0.73rem;color:rgba(150,190,240,0.52);">{desc}</div></div>'
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


def build_feature_vector(process_key, C, Si, Mn, P, S, Ni, Cr, Cu, Mo,
                          ht_temp, soak_time, cool_medium,
                          t_temp=0.0, t_time=0.0):
    CE  = calc_carbon_equiv(C, Mn, Si, Ni, Cr, Mo, Cu)
    A3  = calc_a3(C, Mn, Si, Ni, Cr, Mo)
    HJ  = calc_hollomon_jaffe(t_temp, t_time)
    feat = {
        "C": C, "Si": Si, "Mn": Mn, "P": P, "S": S,
        "Ni": Ni, "Cr": Cr, "Cu": Cu, "Mo": Mo,
        "Carbon_Equiv": CE, "A3_Temp_C": A3, "Delta_HT_A3": ht_temp - A3,
        "Hollomon_Jaffe": HJ, "C_x_Cr": C * Cr,
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
    plot_bgcolor ="rgba(4,7,18,1)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#b0cef0", family="Inter, sans-serif", size=11),
)


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


def _hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def build_tt_profile(process_key, ht_temp, t_temp, soak_time, t_time):
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
    A3_val = calc_a3(0.35)
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
def build_combined_animation(process_key, C, ht_temp, soak_time, cool_medium,
                              t_temp, t_time):
    from PIL import Image as PILImage
    from io import BytesIO
    from scipy.spatial import cKDTree

    # Physical setup
    A1    = 723.0
    Ms    = max(80.0, 539 - 423*C - 30.4*0.85 - 12.1*1.05 - 7.5*0.2)
    COOL  = {"Water":3.0,"Polymer":2.2,"Oil":1.8,"Salt Bath":1.1,"Air":0.4,"Furnace":0.08}
    spd   = COOL.get(cool_medium, 1.0)
    K_gg  = 0.9 * np.exp(-20000 / (ht_temp + 273.15)) * (soak_time / 60.0)
    n_ini = 100
    n_soak= max(16, int(n_ini / (1.0 + 7.0 * K_gg)))

    # Phase colour palette (linear 0-1 RGB)
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

    # Pre-compute grain structures
    seed0 = (int(abs(C * 100)) + int(ht_temp)) % 9973
    GS = {
        "initial": _compute_voronoi_grains(n_ini,     seed=seed0),
        "soaked":  _compute_voronoi_grains(n_soak,    seed=seed0+1),
        "recryst": _compute_voronoi_grains(n_soak+28, seed=seed0+2),
    }
    rng0 = np.random.default_rng(seed0)
    ORI  = {k: rng0.random(len(v[1])) for k, v in GS.items()}

    # Pre-rasterize grain index maps using cKDTree
    IMG  = 140
    gx, gy = np.meshgrid(np.linspace(0, 1, IMG), np.linspace(0, 1, IMG))
    pixels = np.column_stack([gx.ravel(), gy.ravel()])
    GIDX = {}
    for k, (pts, _) in GS.items():
        _, idx  = cKDTree(pts).query(pixels)
        GIDX[k] = idx.clip(0, len(pts)-1)

    # T-t profile
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
            if f < 0.22: return "Heating"
            if f < 0.42: return "Austenitizing"
            return "Air Cooling + Recrystallization"
        elif process_key == "Annealing":
            if f < 0.20: return "Heating"
            if f < 0.45: return "Austenitizing"
            return "Furnace Cooling + Recrystallization"
        else:
            if f < 0.22: return "Heating"
            if f < 0.68: return "Soaking (Sub-A1)"
            return "Air Cooling"

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
                mf = max(0.0,min(1.0,(Ms+60-T)/(Ms+60-30))) if spd>=1.0 else max(0.0,min(0.4,prog*0.4))
                return "soaked",[("martensite",mf),("austenite",1-mf)],30,mf
            if f < 0.65:
                return "soaked",[("martensite",1.0)],35,0.8
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
            gs   = "recryst" if X_rx>0.45 else "soaked"
            fp   = min(0.65,prog*0.65); fe=min(0.35,prog*0.35)
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
            gs   = "recryst" if X_rx>0.40 else "soaked"
            cp   = min(0.68,prog*0.68); fe=min(0.30,prog*0.30)
            return gs,[("coarse_pearlite",cp),("ferrite",fe),("austenite",max(0,1-cp-fe))],60,0.0
        else:
            return "initial",[("ferrite",0.55),("pearlite",0.45)],70,0.0

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
        # Grain boundaries
        gmap = gidx.reshape(IMG, IMG)
        bnd  = np.zeros((IMG, IMG), dtype=bool)
        bnd[:-1]   |= (gmap[:-1]   != gmap[1:])
        bnd[:, :-1] |= (gmap[:, :-1] != gmap[:, 1:])
        px_rgb[bnd] *= 0.14
        # Martensite needles
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

    # Build static phase diagram portion of the figure
    A1_=723; A3_FE=912; PERIT=1495; MELT=1538; EUTE=1147
    C_EUT=0.76; C_AUS=2.11; MAX_C=2.3; C_PER=0.17

    def _a3(c):  return A3_FE - (A3_FE-A1_)/C_EUT * min(c, C_EUT)
    def _acm(c): return A1_ + (EUTE-A1_)/(C_AUS-C_EUT) * (c-C_EUT)

    fig = go.Figure()

    # Phase regions
    for xs, ys, fill, lbl in [
        ([0, 0.022, 0.022, 0],          [0,0,A1_,A1_],                    "rgba(80,200,120,0.20)", "Ferrite"),
        ([0.022,C_EUT,C_EUT,0.022],     [0,0,A1_,A1_],                    "rgba(86,180,211,0.20)", "Ferrite+Pearlite"),
        ([C_EUT,MAX_C,MAX_C,C_EUT],     [0,0,A1_,A1_],                    "rgba(176,122,214,0.20)", "Pearlite+Fe3C"),
        ([0,C_EUT,0],                   [A3_FE,A1_,A1_],                   "rgba(255,140,48,0.20)", "gamma+alpha"),
        ([C_EUT,MAX_C,MAX_C,C_AUS],     [A1_,A1_,EUTE,EUTE],              "rgba(255,92,106,0.20)", "gamma+Fe3C"),
        ([0,C_PER,C_AUS,C_EUT,0],       [MELT,PERIT,EUTE,A1_,A3_FE],     "rgba(255,215,0,0.12)", "Austenite"),
        ([0,MAX_C,MAX_C,C_AUS,C_PER,0], [MELT,EUTE,1620,1620,PERIT,MELT], "rgba(240,230,140,0.11)", "Liquid"),
        ([0,C_PER,C_PER,0],             [MELT,PERIT,1570,1570],            "rgba(180,225,255,0.20)", "delta-Ferrite"),
    ]:
        fig.add_trace(go.Scatter(
            x=xs, y=ys, fill="toself", fillcolor=fill,
            line=dict(width=0.5, color="rgba(255,255,255,0.10)"),
            mode="lines",
            hovertemplate=f"<b>{lbl}</b><br>C=%{{x:.3f}}%  T=%{{y:.0f}}C<extra></extra>",
            showlegend=False,
        ))

    # A3, Acm, A1 lines
    a3_arr = np.linspace(0, C_EUT, 120)
    fig.add_trace(go.Scatter(x=a3_arr, y=[_a3(c) for c in a3_arr],
        line=dict(color="#f0a030", width=1.8), mode="lines", name="A3",
        hovertemplate="A3=%{y:.0f}C<extra></extra>"))
    acm_arr = np.linspace(C_EUT, MAX_C, 120)
    fig.add_trace(go.Scatter(x=acm_arr, y=[_acm(c) for c in acm_arr],
        line=dict(color="#e05050", width=1.8), mode="lines", name="Acm",
        hovertemplate="Acm=%{y:.0f}C<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0,MAX_C], y=[A1_,A1_],
        line=dict(color="#80aaee", width=1.4, dash="dot"),
        mode="lines", name="A1=723C"))

    # Ms line
    Ms_approx = max(80.0, 539 - 423*C)
    fig.add_trace(go.Scatter(x=[0,MAX_C], y=[Ms_approx,Ms_approx],
        line=dict(color="rgba(130,110,255,0.55)", width=1.2, dash="dashdot"),
        mode="lines+text",
        text=["", f"Ms={Ms_approx:.0f}C"],
        textposition="middle right", textfont=dict(size=8, color="#9090ff"),
        name="Ms", showlegend=False))

    # Marker points
    fig.add_trace(go.Scatter(x=[C_EUT], y=[A1_], mode="markers",
        marker=dict(size=8,color="#ffd700",symbol="diamond",line=dict(color="white",width=1.5)),
        name="Eutectoid S", showlegend=False))
    fig.add_trace(go.Scatter(x=[C_PER], y=[PERIT], mode="markers",
        marker=dict(size=8,color="#60c4ff",symbol="triangle-up",line=dict(color="white",width=1.5)),
        name="Peritectic P", showlegend=False))

    # Process + carbon lines (as scatter, safe with domain)
    fig.add_trace(go.Scatter(x=[0,MAX_C], y=[ht_temp,ht_temp],
        line=dict(color="#00b4ff", width=1.5, dash="dashdot"),
        mode="lines+text", text=["", f"{ht_temp:.0f}C"],
        textposition="middle right", textfont=dict(size=8, color="#00b4ff"),
        name="Austenitize", showlegend=False))
    if t_temp > 0:
        fig.add_trace(go.Scatter(x=[0,MAX_C], y=[t_temp,t_temp],
            line=dict(color="#f0a030", width=1.2, dash="dashdot"),
            mode="lines+text", text=["", f"{t_temp:.0f}C"],
            textposition="middle right", textfont=dict(size=8, color="#f0a030"),
            name="Temper", showlegend=False))
    fig.add_trace(go.Scatter(x=[C,C], y=[0,1620],
        line=dict(color="rgba(80,227,194,0.55)", width=1.4, dash="dot"),
        mode="lines+text", text=[f"C={C:.2f}%", ""],
        textposition="top right", textfont=dict(size=8, color="#50e3c2"),
        name="C content", showlegend=False))

    # Full T-t trajectory (faint path at x=C)
    N_PATH   = 80
    path_f   = np.linspace(0, 1, N_PATH)
    path_T   = [T_at(f_) for f_ in path_f]
    fig.add_trace(go.Scatter(
        x=[C]*N_PATH, y=path_T, mode="lines",
        line=dict(color="rgba(255,107,53,0.28)", width=2.0),
        name="HT path", showlegend=False, hoverinfo="skip"))

    # Animated operating point (initial position at frame 0)
    T_init = T_at(0)
    fig.add_trace(go.Scatter(
        x=[C], y=[T_init], mode="markers",
        marker=dict(size=14, color="#ff6b35", symbol="circle",
                    line=dict(color="white", width=2.2)),
        name="Operating point",
        hovertemplate="T=%{y:.0f}C<extra></extra>"))
    N_ANIM = len(fig.data) - 1   # index of the only animated trace

    # Phase region labels (use add_annotation — won't be overwritten by frames)
    for lx, ly, ltxt in [
        (0.011,310,"a"),(0.38,310,"a+P"),(1.55,310,"P+Fe3C"),
        (0.22,780,"g+a"),(1.82,900,"g+Fe3C"),(0.55,1075,"g"),
        (1.05,1390,"L"),(0.05,1510,"d"),
    ]:
        fig.add_annotation(x=lx, y=ly, text=ltxt, showarrow=False,
                           font=dict(size=9,color="rgba(200,220,255,0.52)"),
                           bgcolor="rgba(0,0,0,0)", xref="x", yref="y")

    # Panel title annotations (paper coords — won't be overwritten by frames)
    fig.add_annotation(x=0.235, y=1.03, xref="paper", yref="paper",
                       text="<b>Fe-C Phase Diagram</b>",
                       font=dict(size=12, color="#b0cef0"),
                       showarrow=False, xanchor="center")
    fig.add_annotation(x=0.765, y=1.03, xref="paper", yref="paper",
                       text="<b>Microstructure Evolution</b>",
                       font=dict(size=12, color="#b0cef0"),
                       showarrow=False, xanchor="center")

    # Vertical divider
    fig.add_shape(type="line",
                  x0=0.49, y0=0.03, x1=0.49, y1=1.00,
                  xref="paper", yref="paper",
                  line=dict(color="rgba(0,150,255,0.18)", width=1))

    # Restrict x/y axes to left 47% of figure
    fig.update_layout(
        xaxis=dict(domain=[0.0, 0.46], title="Carbon Content (wt%)", range=[0, MAX_C],
                   gridcolor="rgba(180,210,255,0.06)", zeroline=False,
                   tickfont=dict(size=9)),
        yaxis=dict(domain=[0.10, 0.97], title="Temperature (C)", range=[0, 1640],
                   gridcolor="rgba(180,210,255,0.06)", zeroline=False,
                   tickfont=dict(size=9)),
    )

    # Generate frames
    N_FRAMES = 32
    fracs    = np.linspace(0, 1, N_FRAMES)
    frames   = []

    # Build frame-0 image for the initial layout display
    gs0, pf0, so0, nf0 = frame_state(fracs[0])
    arr0     = make_micro_array(gs0, pf0, so0, nf0)
    img0_src = arr_to_b64(arr0)

    for fi, f in enumerate(fracs):
        T     = T_at(f)
        stage = sname(f)
        gs_key, phase_fracs, seed_off, needle_frac = frame_state(f)
        arr     = make_micro_array(gs_key, phase_fracs, seed_off, needle_frac)
        img_src = arr_to_b64(arr)
        dom_phase = phase_fracs[0][0].replace("_", " ").title()
        n_vis     = sum(1 for p in GS[gs_key][1] if p is not None)
        frame = go.Frame(
            data=[go.Scatter(x=[C], y=[T])],
            traces=[N_ANIM],
            layout=go.Layout(
                title=dict(text=(
                    f"<b>{stage}</b>  |  T = {T:.0f} C  |  "
                    f"{dom_phase}  |  ~{n_vis} grains"
                )),
                images=[dict(
                    source=img_src,
                    xref="paper", yref="paper",
                    x=0.535, y=0.97,
                    sizex=0.45, sizey=0.87,
                    sizing="stretch", layer="above",
                )],
            ),
            name=str(fi),
        )
        frames.append(frame)

    fig.frames = frames

    # Animation controls
    fig.update_layout(
        updatemenus=[dict(
            type="buttons", showactive=False,
            x=0.53, y=-0.04, xanchor="left", yanchor="top",
            bgcolor="rgba(6,12,32,0.92)",
            bordercolor="rgba(0,150,255,0.30)",
            font=dict(color="#b0cef0", size=12),
            buttons=[
                dict(label="Play",
                     method="animate",
                     args=[None, dict(frame=dict(duration=110, redraw=True),
                                      fromcurrent=True, mode="immediate")]),
                dict(label="Pause",
                     method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate")]),
            ],
        )],
        sliders=[dict(
            active=0, x=0.0, y=-0.06, len=1.0,
            currentvalue=dict(visible=False),
            bgcolor="rgba(6,12,32,0.80)",
            bordercolor="rgba(0,150,255,0.18)",
            tickcolor="rgba(0,150,255,0.35)",
            steps=[dict(
                args=[[str(i)], dict(frame=dict(duration=80, redraw=True),
                                      mode="immediate")],
                method="animate", label="",
            ) for i in range(N_FRAMES)],
            pad=dict(t=12, b=8),
        )],
    )

    # Final layout settings
    fig.update_layout(
        **_BASE,
        title=dict(
            text=(f"<b>Heating</b>  |  T = {T_at(0):.0f} C  |  "
                  f"Ferrite + Pearlite  |  ~{n_ini} grains"),
            font=dict(size=12), x=0.50, xanchor="center", y=0.985,
        ),
        images=[dict(
            source=img0_src,
            xref="paper", yref="paper",
            x=0.535, y=0.97, sizex=0.45, sizey=0.87,
            sizing="stretch", layer="above",
        )],
        legend=dict(bgcolor="rgba(4,7,18,0.82)", bordercolor="rgba(0,150,255,0.20)",
                    borderwidth=1, font=dict(size=8),
                    x=0.01, y=0.99, xanchor="left", yanchor="top"),
        height=590,
        margin=dict(l=50, r=20, t=50, b=88),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div style="text-align:center;padding:2.2rem 0 1.2rem;">'
    '<div class="miq-hero-title" style="font-family:\'Rajdhani\',sans-serif;font-size:3.4rem;'
    'font-weight:700;background:linear-gradient(135deg,#0099ff 0%,#55c8ff 42%,#f0a030 100%);'
    '-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
    'letter-spacing:0.07em;line-height:1;">MAST IQ</div>'
    '<div class="miq-hero-sub" style="color:rgba(150,190,240,0.48);font-size:0.84rem;'
    'letter-spacing:0.24em;text-transform:uppercase;margin-top:7px;">'
    'Alloy Intelligence  &middot;  Materials Advanced Science &amp; Technology</div>'
    '<div style="width:110px;height:2px;'
    'background:linear-gradient(90deg,transparent,#0099ff,#f0a030,transparent);'
    'margin:14px auto 0;"></div>'
    '</div>',
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — INPUTS
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:6px 0 10px;">'
        '<span style="font-family:\'Rajdhani\',sans-serif;font-size:1.22rem;'
        'font-weight:700;background:linear-gradient(90deg,#ff7b2e,#3db8ff);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
        'letter-spacing:0.10em;">INPUT PARAMETERS</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Theme toggle — session_state.miq_theme_cryo is read by CSS block at top of script
    st.toggle("❄  Cryo visual mode",
              key="miq_theme_cryo",
              help="Switch between Forge (warm ember) and Cryo (cool plasma) accent themes.")

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
    st.markdown(
        f'<div style="background:rgba(0,100,200,0.10);border:1px solid rgba(0,150,255,0.22);'
        f'border-radius:8px;padding:8px 12px;margin:6px 0;">'
        f'<span style="font-size:0.76rem;color:rgba(150,190,240,0.58);">Carbon Equiv (IIW): </span>'
        f'<span style="font-family:\'JetBrains Mono\',monospace;color:#4c9ed0;font-weight:600;">{CE:.4f}</span><br>'
        f'<span style="font-size:0.76rem;color:rgba(150,190,240,0.58);">A3 Temperature: </span>'
        f'<span style="font-family:\'JetBrains Mono\',monospace;color:#f0a030;font-weight:600;">{A3v:.0f} C</span>'
        f'</div>',
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
            f'<div style="background:rgba(100,60,180,0.10);border:1px solid rgba(120,80,200,0.22);'
            f'border-radius:8px;padding:7px 12px;margin-top:6px;">'
            f'<span style="font-size:0.74rem;color:rgba(150,190,240,0.58);">Hollomon-Jaffe H: </span>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;color:#9370db;font-weight:600;">{HJ:,.0f}</span>'
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
    '<div class="miq-mobile-hint">'
    '<span style="font-size:1.1rem;">☰</span>'
    '<span>Tap the menu icon (top-left) to open Input Parameters.</span>'
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
                f'<div style="background:rgba(6,12,34,0.75);border:1px solid rgba(0,150,255,0.18);'
                f'border-radius:10px;padding:16px;">'
                f'<div style="font-family:\'Rajdhani\',sans-serif;font-size:1.0rem;'
                f'color:#0099ff;font-weight:700;margin-bottom:8px;">{num} — {ttl}</div>'
                f'<div style="font-size:0.78rem;color:rgba(150,190,240,0.55);">{body}</div>'
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
    st.markdown(
        f'<div style="background:linear-gradient(135deg,{pc}15,rgba(4,7,18,0.75));'
        f'border:1px solid {pc}40;border-left:4px solid {pc};'
        f'border-radius:12px;padding:13px 18px;margin-bottom:16px;'
        f'display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">'
        f'<div>'
        f'<span style="font-family:\'Rajdhani\',sans-serif;font-size:1.25rem;font-weight:700;color:{pc};">'
        f'{plabel}</span>'
        f'<span style="font-size:0.76rem;color:rgba(150,190,240,0.48);margin-left:12px;">'
        f'{ht_s:.0f}C  {sk_s:.0f} min  {cool_s}{temper_str}</span>'
        f'</div>'
        f'<div style="display:flex;gap:8px;flex-wrap:wrap;">'
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
        '<div style="background:rgba(0,80,160,0.10);border:1px solid rgba(0,150,255,0.18);'
        'border-radius:8px;padding:9px 14px;margin-bottom:14px;font-size:0.76rem;'
        'color:rgba(150,190,240,0.60);">'
        'Model disclaimer: Predictions are ML estimates from curated heat treatment datasets. '
        'For safety-critical or industrial certification purposes, validate against '
        'standardised material testing (ASTM / ISO). Accuracy may vary outside training range.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "⟐ Analysis",
        "⬢ Phase Diagram",
        "⎍ T-t Profile",
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

    # Tab 2: Enhanced Phase Diagram
    with tab2:
        st.plotly_chart(
            build_phase_diagram(C_s, ht_s, tt_s, pkey, cool_s,
                                 Mn=Mn_s, Cr=Cr_s, Mo=Mo_s),
            width='stretch',
        )
        st.caption(
            "Operating point marks the austenitizing condition. "
            "Arrows show heating (orange) and cooling (coloured by medium) paths. "
            "Ms and Bs lines are composition-specific approximations. "
            "The diagram shows equilibrium phases; actual microstructure depends on cooling rate."
        )

    # Tab 3: T-t Profile
    with tab3:
        st.plotly_chart(
            build_tt_profile(pkey, ht_s, tt_s, sk_s, ttime_s),
            width='stretch',
        )
        st.caption("Schematic temperature-time profile (times scaled for illustration). "
                   "A1 and A3 reference lines shown where applicable.")

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
                "Phase Diagram and Microstructure Simulation",
                "Interactive animation: Fe-C operating path + EBSD-style grain evolution side by side.",
                "🎬",
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
            with st.spinner("Building interactive simulation — this may take up to 20 s..."):
                anim_fig = build_combined_animation(*sim_p)
            st.plotly_chart(anim_fig, width='stretch')

            st.markdown(
                '<div style="background:rgba(4,7,18,0.75);border:1px solid rgba(0,150,255,0.20);'
                'border-radius:10px;padding:10px 16px;margin-top:8px;font-size:0.79rem;'
                'color:rgba(150,190,240,0.60);">'
                'Microstructure colour key:  '
                'Yellow = Austenite  |  Green = Ferrite  |  Brown = Pearlite  |  '
                'Dark blue = Martensite  |  Light blue = Tempered Martensite  |  '
                'Purple = Bainite  —  '
                'Dark lines = grain boundaries   Bright streaks = martensite laths'
                '</div>',
                unsafe_allow_html=True,
            )

            # Physics expander
            _proc, _C, _ht, _soak, _cool, _tt, _ttime = sim_p
            Ms_v = max(80.0, 539 - 423*_C - 30.4*0.85 - 12.1*1.05 - 7.5*0.2)
            K_gg = 0.9 * np.exp(-20000 / (_ht + 273.15)) * (_soak / 60.0)
            n_s  = max(16, int(100 / (1.0 + 7.0 * K_gg)))
            with st.expander("Simulation physics parameters", expanded=False):
                phys_rows = [
                    ("Martensite start Ms",        f"{Ms_v:.0f}",                  "C"),
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
