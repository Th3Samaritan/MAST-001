"""
Steel Heat Treatment Mechanical Property Predictor
====================================================
Professional UI  —  XGBoost multi-output regression
Predicts: Tensile Strength · Hardness · Fatigue Strength
"""

import os
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
    page_title="SteelSight — Mechanical Property Predictor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  DESIGN SYSTEM  ─  all styling in one place
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & base ──────────────────────────────────────────────────────── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }

.stApp {
    background: radial-gradient(ellipse at 20% 0%, #0d1f3c 0%, #06090f 50%, #0a0d1a 100%);
    min-height: 100vh;
}

/* ── Sidebar ───────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080d1e 0%, #0b1224 60%, #06090f 100%) !important;
    border-right: 1px solid rgba(76,114,176,0.25) !important;
}
[data-testid="stSidebar"] .block-container { padding: 1rem !important; }
[data-testid="stSidebar"] label {
    color: rgba(200,215,255,0.75) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
}

/* ── Number inputs ─────────────────────────────────────────────────────── */
.stNumberInput input {
    background: rgba(13,25,50,0.85) !important;
    border: 1px solid rgba(76,114,176,0.30) !important;
    border-radius: 8px !important;
    color: #e8f0ff !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.88rem !important;
    transition: border-color 0.2s !important;
}
.stNumberInput input:focus {
    border-color: rgba(76,114,176,0.8) !important;
    box-shadow: 0 0 0 2px rgba(76,114,176,0.20) !important;
}
.stNumberInput button {
    background: rgba(13,25,50,0.85) !important;
    border: 1px solid rgba(76,114,176,0.25) !important;
    color: #7b9fd4 !important;
}

/* ── Sliders ───────────────────────────────────────────────────────────── */
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #4c72b0, #00b4d8) !important;
}
[data-testid="stSlider"] [role="slider"] {
    background: white !important;
    border: 2px solid #4c72b0 !important;
    box-shadow: 0 0 8px rgba(76,114,176,0.6) !important;
}
[data-testid="stSlider"] p {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    color: #7b9fd4 !important;
}

/* ── Primary button ────────────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1a3a6b 0%, #2557ab 50%, #1e4d9e 100%) !important;
    border: 1px solid rgba(76,114,176,0.55) !important;
    color: #e8f4ff !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.65rem 1.5rem !important;
    box-shadow: 0 4px 20px rgba(37,87,171,0.45), 0 0 0 1px rgba(255,255,255,0.05) inset !important;
    transition: all 0.25s ease !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #2557ab 0%, #3473c7 50%, #2a63b5 100%) !important;
    box-shadow: 0 6px 30px rgba(52,115,199,0.60), 0 0 25px rgba(76,114,176,0.35) !important;
    transform: translateY(-2px) !important;
}

/* ── Tabs ──────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(8,13,30,0.8) !important;
    border: 1px solid rgba(76,114,176,0.2) !important;
    border-radius: 12px !important;
    padding: 5px !important;
    gap: 3px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: rgba(180,200,240,0.55) !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.45rem 1.4rem !important;
    transition: all 0.2s !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, rgba(76,114,176,0.30), rgba(0,180,216,0.15)) !important;
    color: #e8f0ff !important;
    box-shadow: 0 2px 12px rgba(76,114,176,0.25) !important;
}

/* ── Expanders ─────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: rgba(8,13,30,0.6) !important;
    border: 1px solid rgba(76,114,176,0.18) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    color: rgba(180,200,240,0.8) !important;
    font-size: 0.85rem !important;
}

/* ── Dataframe ─────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(76,114,176,0.2) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ── Divider ───────────────────────────────────────────────────────────── */
hr { border-color: rgba(76,114,176,0.20) !important; }

/* ── Block container ───────────────────────────────────────────────────── */
.block-container { padding: 1.8rem 2.5rem 1rem !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HTML COMPONENT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def html_metric_card(title, value, unit, color, icon, subtitle=""):
    glow = color.replace("#", "")
    return f"""
    <div style="
        background: linear-gradient(145deg, rgba(13,25,50,0.95) 0%, rgba(8,13,30,0.98) 100%);
        border: 1px solid {color}55;
        border-top: 3px solid {color};
        border-radius: 14px;
        padding: 22px 20px 18px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.45), 0 0 0 1px rgba(255,255,255,0.03) inset,
                    0 0 30px {color}18;
        position: relative;
        overflow: hidden;
    ">
      <div style="position:absolute;top:-30px;right:-20px;font-size:6rem;
                  opacity:0.04;user-select:none;">{icon}</div>
      <div style="font-size:2.2rem;margin-bottom:6px;">{icon}</div>
      <div style="
          font-size: 2.6rem;
          font-weight: 700;
          color: {color};
          font-family: 'Rajdhani', sans-serif;
          line-height: 1;
          letter-spacing: -0.02em;
      ">{value}</div>
      <div style="
          font-size: 0.7rem;
          color: rgba(180,200,255,0.45);
          text-transform: uppercase;
          letter-spacing: 0.14em;
          margin-top: 10px;
          font-weight: 600;
      ">{title}</div>
      <div style="font-size:0.78rem;color:{color}99;margin-top:3px;">{unit}</div>
      {f'<div style="font-size:0.72rem;color:rgba(180,200,255,0.35);margin-top:6px;">{subtitle}</div>' if subtitle else ''}
    </div>"""


def html_section_header(title, subtitle="", icon=""):
    return f"""
    <div style="margin: 0 0 18px 0;">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
        <span style="font-size:1.3rem;">{icon}</span>
        <span style="
            font-family:'Rajdhani',sans-serif;
            font-size:1.35rem;
            font-weight:700;
            color:#e8f0ff;
            letter-spacing:0.03em;
        ">{title}</span>
      </div>
      {f'<p style="color:rgba(180,200,255,0.50);font-size:0.82rem;margin:0 0 0 2.2rem;">{subtitle}</p>' if subtitle else ''}
    </div>"""


def html_badge(text, color, bg_opacity=0.2):
    return (f'<span style="display:inline-block;padding:3px 10px;border-radius:20px;'
            f'background:{color}{int(bg_opacity*255):02x};border:1px solid {color}66;'
            f'color:{color};font-size:0.72rem;font-weight:600;letter-spacing:0.06em;'
            f'text-transform:uppercase;">{text}</span>')


def html_stat_pill(label, value, color="#4c72b0"):
    return f"""
    <div style="
        display:inline-flex;align-items:center;gap:8px;
        background:rgba(13,25,50,0.85);
        border:1px solid {color}40;border-radius:30px;
        padding:6px 14px;margin:4px;
    ">
      <span style="width:7px;height:7px;border-radius:50%;background:{color};
                   box-shadow:0 0 6px {color};display:inline-block;"></span>
      <span style="font-size:0.78rem;color:rgba(180,200,255,0.6);">{label}</span>
      <span style="font-size:0.82rem;font-weight:600;color:{color};
                   font-family:'JetBrains Mono',monospace;">{value}</span>
    </div>"""


def html_phase_badge(phase):
    """Color-coded phase badge."""
    colors = {
        "Austenite (γ)":                  ("#ffd700", "🟡"),
        "Austenite (γ) + Ferrite (α)":    ("#ff8c30", "🟠"),
        "Austenite (γ) + Cementite (Fe₃C)":("#ff5c6a","🔴"),
        "Ferrite (α)":                    ("#50c878", "🟢"),
        "Ferrite (α) + Pearlite":         ("#56b4d3", "🔵"),
        "Pearlite + Cementite (Fe₃C)":    ("#b07ad6", "🟣"),
        "Liquid":                         ("#f0e68c", "⚪"),
    }
    color, dot = colors.get(phase, ("#aaa", "⚫"))
    return (f'<span style="display:inline-flex;align-items:center;gap:6px;'
            f'background:{color}18;border:1px solid {color}50;border-radius:6px;'
            f'padding:4px 10px;color:{color};font-size:0.80rem;font-weight:500;">'
            f'{dot} {phase}</span>')


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL  (v2: per-target models  ·  v1 fallback: single multi-output model)
# ══════════════════════════════════════════════════════════════════════════════
import json as json_lib

TARGETS = ["Tensile", "Hardness", "Fatigue"]
BASE_FEATURES = ["C","Si","Mn","P","S","Ni","Cr","Cu","Mo","NT","TT","QT"]

@st.cache_resource(show_spinner="Loading model…")
def load_models():
    """Try v2 per-target models first, fall back to v1 single model."""
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    metrics = None

    # ── Try v2 per-target models ──────────────────────────────────────
    v2_paths = {t: os.path.join(models_dir, f"xgb_{t.lower()}.json") for t in TARGETS}
    if all(os.path.exists(p) for p in v2_paths.values()):
        per_target = {}
        for t, p in v2_paths.items():
            m = XGBRegressor()
            m.load_model(p)
            per_target[t] = m
        # Load metrics if available
        metrics_path = os.path.join(models_dir, "model_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json_lib.load(f)
        return {"mode": "v2", "models": per_target, "metrics": metrics}

    # ── Fallback: v1 single model ─────────────────────────────────────
    if os.path.exists("xgb_model.json"):
        m = XGBRegressor(); m.load_model("xgb_model.json")
        return {"mode": "v1", "models": m, "metrics": None}
    if os.path.exists("best_xgb_model.pkl"):
        import joblib
        return {"mode": "v1", "models": joblib.load("best_xgb_model.pkl"), "metrics": None}

    st.error("⚠️  Model file not found. Run the training notebook first.")
    st.stop()

model_pack    = load_models()
MODEL_MODE    = model_pack["mode"]
MODEL_METRICS = model_pack["metrics"]

# Feature order depends on model version
if MODEL_MODE == "v2" and MODEL_METRICS and "features" in MODEL_METRICS:
    FEATURE_ORDER = MODEL_METRICS["features"]
else:
    FEATURE_ORDER = BASE_FEATURES

def add_engineered_features(inp):
    """Add CE, DeltaT, C_x_Cr if using v2 models."""
    if MODEL_MODE != "v2":
        return inp
    inp["CE"]     = inp["C"] + inp["Mn"]/6 + (inp["Cr"]+inp["Mo"])/5 + (inp["Ni"]+inp["Cu"])/15
    inp["DeltaT"] = inp["NT"] - inp["TT"]
    inp["C_x_Cr"] = inp["C"] * inp["Cr"]
    return inp

def predict_properties(inp_dict):
    """Predict using v2 per-target or v1 multi-output model."""
    inp = add_engineered_features(inp_dict.copy())
    df  = pd.DataFrame([inp])[FEATURE_ORDER]

    if MODEL_MODE == "v2":
        models = model_pack["models"]
        tensile  = max(0.0, float(models["Tensile"].predict(df)[0]))
        hardness = max(0.0, float(models["Hardness"].predict(df)[0]))
        fatigue  = max(0.0, float(models["Fatigue"].predict(df)[0]))
    else:
        raw = model_pack["models"].predict(df)
        tensile  = max(0.0, float(raw[0][0]))
        hardness = max(0.0, float(raw[0][1]))
        fatigue  = max(0.0, float(raw[0][2]))
    return tensile, hardness, fatigue

# Load training ranges from metrics or use defaults
if MODEL_METRICS and "training_ranges" in MODEL_METRICS:
    TRAINING_RANGES = {k: (v["min"], v["max"]) for k, v in MODEL_METRICS["training_ranges"].items()}
else:
    TRAINING_RANGES = {
        "NT":(825,900), "QT":(825,870), "TT":(550,680),
        "C":(0.30,0.50), "Si":(0.10,1.50), "Mn":(0.50,1.50),
        "P":(0.005,0.040), "S":(0.005,0.150),
        "Ni":(0.00,0.50), "Cr":(0.00,0.50), "Cu":(0.00,0.50), "Mo":(0.00,0.50),
    }

PROP_RANGES = {
    "Tensile":  (400,  1800),
    "Hardness": (150,  450),
    "Fatigue":  (200,  700),
}


# ══════════════════════════════════════════════════════════════════════════════
#  FE-C PHASE DIAGRAM  CONSTANTS & FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
A1           = 723
A3_PURE_FE   = 912
PERITECTIC_T = 1495
MELT_FE      = 1538
EUTECTIC_T   = 1147
C_FERRITE_MAX = 0.022
C_EUTECTOID   = 0.76
C_AUS_MAX     = 2.11
MAX_C         = 2.3


def a3_temp(c):
    return A3_PURE_FE - (A3_PURE_FE - A1) / C_EUTECTOID * min(c, C_EUTECTOID)

def acm_temp(c):
    return A1 + (EUTECTIC_T - A1) / (C_AUS_MAX - C_EUTECTOID) * (c - C_EUTECTOID)

def get_phase(c, T):
    if T >= 1400:  return "Liquid"
    if T >= A1:
        if c <= C_EUTECTOID and T < a3_temp(c):  return "Austenite (γ) + Ferrite (α)"
        if c > C_EUTECTOID and T < acm_temp(c):  return "Austenite (γ) + Cementite (Fe₃C)"
        return "Austenite (γ)"
    if c <= C_FERRITE_MAX: return "Ferrite (α)"
    if c <= C_EUTECTOID:   return "Ferrite + Pearlite"
    return "Pearlite + Cementite (Fe₃C)"

def check_ood(d):
    return [f"**{k}** = {v} outside [{TRAINING_RANGES[k][0]}, {TRAINING_RANGES[k][1]}]"
            for k, v in d.items()
            if v < TRAINING_RANGES[k][0] or v > TRAINING_RANGES[k][1]]

def steel_grade(tensile):
    if tensile < 600:   return ("Low Strength", "#56b4d3")
    if tensile < 900:   return ("Medium Strength", "#50c878")
    if tensile < 1200:  return ("High Strength",   "#ffd700")
    if tensile < 1500:  return ("Very High Strength","#ff8c30")
    return ("Ultra-High Strength", "#ff5c6a")


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTLY BUILDERS
# ══════════════════════════════════════════════════════════════════════════════
_PLOTLY_BASE = dict(
    plot_bgcolor ="rgba(8,13,30,1)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c8d8ff", family="Inter, sans-serif", size=11),
    margin=dict(l=60, r=20, t=50, b=50),
)


def build_gauge_charts(tensile, hardness, fatigue):
    """Three gauge/indicator charts side by side."""
    specs  = [[{"type":"indicator"}, {"type":"indicator"}, {"type":"indicator"}]]
    fig    = make_subplots(rows=1, cols=3, specs=specs)
    props  = [
        (tensile,  "Tensile Strength", "MPa", PROP_RANGES["Tensile"],  "#4c72b0"),
        (hardness, "Hardness",         "HB",  PROP_RANGES["Hardness"], "#dd8452"),
        (fatigue,  "Fatigue Strength", "MPa", PROP_RANGES["Fatigue"],  "#55a868"),
    ]
    for col_i, (val, title, unit, (lo, hi), col) in enumerate(props, 1):
        mid = (lo + hi) / 2
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=val,
            number=dict(font=dict(size=28, color=col, family="Rajdhani, sans-serif"),
                        suffix=f" {unit}"),
            title=dict(text=f"<b>{title}</b>", font=dict(size=12, color="#c8d8ff")),
            gauge=dict(
                axis=dict(range=[lo, hi], tickwidth=1, tickcolor="rgba(200,216,255,0.3)",
                          tickfont=dict(size=9, color="rgba(200,216,255,0.4)")),
                bar=dict(color=col, thickness=0.25),
                bgcolor="rgba(8,13,30,0)",
                borderwidth=0,
                steps=[
                    dict(range=[lo, lo+(hi-lo)*0.33], color="rgba(255,255,255,0.03)"),
                    dict(range=[lo+(hi-lo)*0.33, lo+(hi-lo)*0.66], color="rgba(255,255,255,0.05)"),
                    dict(range=[lo+(hi-lo)*0.66, hi], color="rgba(255,255,255,0.08)"),
                ],
                threshold=dict(line=dict(color="white", width=3),
                               thickness=0.75, value=val),
            ),
        ), row=1, col=col_i)

    fig.update_layout(**_PLOTLY_BASE)
    fig.update_layout(
        height=260,
        margin=dict(l=30, r=30, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_radar_chart(tensile, hardness, fatigue):
    """Radar/spider chart showing normalized properties."""
    lo_t, hi_t = PROP_RANGES["Tensile"]
    lo_h, hi_h = PROP_RANGES["Hardness"]
    lo_f, hi_f = PROP_RANGES["Fatigue"]

    norm_t = (tensile  - lo_t) / (hi_t - lo_t) * 100
    norm_h = (hardness - lo_h) / (hi_h - lo_h) * 100
    norm_f = (fatigue  - lo_f) / (hi_f - lo_f) * 100

    cats   = ["Tensile Strength", "Hardness", "Fatigue Strength"]
    values = [norm_t, norm_h, norm_f]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=cats + [cats[0]],
        fill="toself",
        fillcolor="rgba(76,114,176,0.25)",
        line=dict(color="#4c72b0", width=2.5),
        marker=dict(size=7, color="#64b5f6"),
        name="Predicted",
    ))
    # Average reference
    fig.add_trace(go.Scatterpolar(
        r=[50, 50, 50, 50],
        theta=cats + [cats[0]],
        fill="toself",
        fillcolor="rgba(255,255,255,0.03)",
        line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dot"),
        name="Dataset avg.",
    ))
    fig.update_layout(**_PLOTLY_BASE)
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(8,13,30,0.5)",
            radialaxis=dict(range=[0,100], ticksuffix="%",
                            gridcolor="rgba(200,216,255,0.10)",
                            linecolor="rgba(200,216,255,0.10)",
                            tickfont=dict(size=8, color="rgba(200,216,255,0.4)")),
            angularaxis=dict(gridcolor="rgba(200,216,255,0.10)",
                             linecolor="rgba(200,216,255,0.15)",
                             tickfont=dict(size=10, color="#c8d8ff")),
        ),
        showlegend=True,
        legend=dict(bgcolor="rgba(8,13,30,0.7)", bordercolor="rgba(76,114,176,0.3)",
                    borderwidth=1, font=dict(size=10)),
        height=320,
        margin=dict(l=40, r=40, t=30, b=20),
    )
    return fig


def build_phase_diagram(C, NT, QT, TT, animated=False):
    """Full interactive Fe-C phase diagram with optional animation."""
    fig  = go.Figure()

    def region(xs, ys, r, g, b, name, a=0.42):
        fig.add_trace(go.Scatter(
            x=list(xs)+[xs[0]], y=list(ys)+[ys[0]],
            fill="toself", fillcolor=f"rgba({r},{g},{b},{a})",
            line=dict(color="rgba(0,0,0,0)", width=0),
            name=name, hoverinfo="name",
            legendgroup="phases", showlegend=True,
        ))

    # ── Phase regions ──────────────────────────────────────────────────────
    region([0,C_FERRITE_MAX,C_FERRITE_MAX,0],      [0,0,A1,A1],           60,200, 80, "Ferrite (α)")
    region([C_FERRITE_MAX,C_EUTECTOID,C_EUTECTOID,C_FERRITE_MAX],
                                                    [0,0,A1,A1],           70,170,220, "Ferrite + Pearlite")
    region([C_EUTECTOID,MAX_C,MAX_C,C_EUTECTOID],  [0,0,A1,A1],          170,110,220, "Pearlite + Cementite")
    region([0,C_EUTECTOID,0],                       [A3_PURE_FE,A1,A1],   255,150, 60, "Austenite + Ferrite")
    region([C_EUTECTOID,C_AUS_MAX,C_AUS_MAX,C_EUTECTOID],
                                                    [A1,EUTECTIC_T,A1,A1],255,100,120, "Austenite + Cementite")
    region([0,C_EUTECTOID,C_AUS_MAX,0.17,0],
           [A3_PURE_FE,A1,EUTECTIC_T,PERITECTIC_T,MELT_FE],              255,210, 60, "Austenite (γ)")
    region([0,0.17,C_AUS_MAX,0],
           [MELT_FE,PERITECTIC_T,EUTECTIC_T,MELT_FE],                    240,235,150, "Liquid / Mushy Zone")

    # ── Boundary lines ─────────────────────────────────────────────────────
    for xs, ys, col, w, dash, name in [
        ([0,C_EUTECTOID],        [A3_PURE_FE,A1],      "#4a90d9",  2.5, "solid", "A₃ line"),
        ([C_EUTECTOID,C_AUS_MAX],[A1,EUTECTIC_T],       "#2ecc71",  2.5, "solid", "Acm line"),
        ([0,MAX_C],              [A1,A1],                "#e74c3c",  2.2, "solid", f"A₁ = {A1}°C"),
        ([0.17,C_AUS_MAX],       [PERITECTIC_T,EUTECTIC_T],"#888",  1.3, "dash",  "Solidus"),
    ]:
        fig.add_trace(go.Scatter(x=xs,y=ys,mode="lines",
            line=dict(color=col,width=w,dash=dash),name=name,legendgroup="lines"))

    # ── Heat treatment guide lines ──────────────────────────────────────────
    for temp, col, lbl in [(NT,"rgba(80,155,255,0.90)",f"Normalizing  {NT}°C"),
                            (QT,"rgba(80,220,130,0.90)",f"Quenching  {QT}°C"),
                            (TT,"rgba(255,185,50,0.90)", f"Tempering  {TT}°C")]:
        fig.add_trace(go.Scatter(x=[0,MAX_C],y=[temp,temp],mode="lines",
            line=dict(color=col,width=1.8,dash="dot"),name=lbl,legendgroup="ht"))
        fig.add_annotation(x=MAX_C-0.04,y=temp+16,text=f"<b>{temp}°C</b>",
            showarrow=False,font=dict(size=10,color=col),xanchor="right")

    # ── Carbon line ─────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(x=[C,C],y=[0,MELT_FE],mode="lines",
        line=dict(color="rgba(230,70,70,0.85)",width=2.2,dash="dash"),
        name=f"C = {C:.3f}%"))

    # ── Eutectoid marker ─────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(x=[C_EUTECTOID],y=[A1],mode="markers+text",
        marker=dict(color="#e74c3c",size=10,symbol="circle"),
        text=["  Eutectoid (S)"],textposition="top right",
        textfont=dict(size=10,color="#e74c3c"),showlegend=False))

    # ── Phase region labels ──────────────────────────────────────────────────
    anns = [
        dict(x=0.009,y=330, text="Ferrite (α)",   font=dict(size=10,color="#4ddb82")),
        dict(x=0.38, y=330, text="α + Pearlite",  font=dict(size=10,color="#56b4d3")),
        dict(x=1.25, y=330, text="Pearlite+Fe₃C", font=dict(size=10,color="#b07ad6")),
        dict(x=0.18, y=815, text="γ + α",         font=dict(size=11,color="#ff8c30")),
        dict(x=1.50, y=870, text="γ + Fe₃C",      font=dict(size=11,color="#ff5c6a")),
        dict(x=0.62, y=1170,text="Austenite (γ)", font=dict(size=14,color="#ffd700",
                                                              family="Rajdhani Black")),
        dict(x=0.15, y=1530,text="Liquid",        font=dict(size=11,color="#c8b830")),
    ]
    for a in anns:
        a.update(showarrow=False, xref="x", yref="y")

    # ── Steel position marker ────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[C], y=[25], mode="markers+text",
        marker=dict(size=18, color="white", symbol="circle",
                    line=dict(color="#4c72b0", width=3),
                    ),
        text=[" 25°C"], textposition="top right",
        textfont=dict(size=11, color="white"),
        name="Steel position", showlegend=True,
    ))
    steel_idx = len(fig.data) - 1

    # ── Layout ──────────────────────────────────────────────────────────────
    b_margin = 150 if animated else 50
    fig.update_layout(**_PLOTLY_BASE)
    fig.update_layout(
        xaxis=dict(title="Carbon Content (wt%)", range=[0,MAX_C],
                   gridcolor="rgba(255,255,255,0.05)", showgrid=True,
                   color="#c8d8ff", zeroline=False),
        yaxis=dict(title="Temperature (°C)", range=[0,1600],
                   gridcolor="rgba(255,255,255,0.05)", showgrid=True,
                   color="#c8d8ff", zeroline=False),
        title=dict(text="Fe-C Phase Diagram  ·  Steel Region  (0 – 2.3 wt% C)",
                   font=dict(size=15, color="white", family="Rajdhani, sans-serif")),
        legend=dict(bgcolor="rgba(8,13,30,0.88)", bordercolor="rgba(76,114,176,0.35)",
                    borderwidth=1, font=dict(size=10),
                    tracegroupgap=4, groupclick="toggleitem"),
        annotations=anns,
        height=650,
        margin=dict(l=65, r=20, t=58, b=b_margin),
    )

    if not animated:
        return fig

    # ══════════════════════════════════════════════════════════════════════
    #  ANIMATION PATH
    # ══════════════════════════════════════════════════════════════════════
    heat_nt = np.linspace(25,  NT, 28).tolist()
    hold_nt = [float(NT)] * 10
    quench  = np.linspace(NT,  25, 22).tolist()
    heat_tt = np.linspace(25,  TT, 20).tolist()
    hold_tt = [float(TT)] * 10
    cool_rt = np.linspace(TT,  25, 16).tolist()
    path    = heat_nt + hold_nt + quench + heat_tt + hold_tt + cool_rt

    b0 = len(heat_nt)
    b1 = b0 + len(hold_nt)
    b2 = b1 + len(quench)
    b3 = b2 + len(heat_tt)
    b4 = b3 + len(hold_tt)

    STAGE_INFO = [
        (b0,  "🔥 Heating to Normalising",
               "Steel heats through ferrite+pearlite into the two‑phase and then fully austenitic region."),
        (b1,  "⏳ Normalising Hold",
               "Fully austenitic — carbides dissolved, grain structure homogenised and refined."),
        (b2,  "💧 Quenching  (rapid cooling)",
               "Austenite is 'frozen' by fast cooling → martensite (non‑equilibrium phase, very hard)."),
        (b3,  "🔥 Heating to Tempering Temperature",
               "Steel reheated below A₁ to relieve internal stresses; no phase change on equilibrium diagram."),
        (b4,  "⏳ Tempering Hold",
               "Martensite → tempered martensite: toughness and ductility improve significantly."),
        (9999,"❄️ Cooling to Room Temperature",
               "Final microstructure: tempered martensite — hard, tough, and with excellent fatigue life."),
    ]

    def stage_info(i):
        for bound, title, desc in STAGE_INFO:
            if i < bound:
                return title, desc
        return STAGE_INFO[-1][1], STAGE_INFO[-1][2]

    frames = []
    for i, T in enumerate(path):
        title, desc = stage_info(i)
        phase = get_phase(C, T)
        lbl = (f"<b>{title}</b>   ·   <span style='color:#64b5f6'>{T:.0f}°C</span><br>"
               f"<span style='color:#aaa;font-size:11px'>Phase: {phase}</span><br>"
               f"<span style='color:#666;font-size:10px'>{desc}</span>")

        frame_anns = anns + [{
            "text": lbl, "xref":"paper","yref":"paper",
            "x":0.50, "y":-0.19, "showarrow":False, "align":"center",
            "font":{"size":12,"color":"white"},
            "bgcolor":"rgba(8,13,30,0.94)",
            "bordercolor":"rgba(76,114,176,0.45)",
            "borderwidth":1, "borderpad":10,
        }]

        frames.append(go.Frame(
            data=[go.Scatter(
                x=[C], y=[T], mode="markers+text",
                marker=dict(size=18, color="white", symbol="circle",
                            line=dict(color="#4c72b0", width=3)),
                text=[f" {T:.0f}°C"], textposition="top right",
                textfont=dict(size=11, color="white"),
            )],
            traces=[steel_idx],
            layout=go.Layout(annotations=frame_anns),
            name=str(i),
        ))

    fig.frames = frames

    ANIM_BUTTON_STYLE = dict(
        bgcolor="rgba(8,13,30,0.90)",
        font=dict(color="white", size=12),
        bordercolor="rgba(76,114,176,0.45)",
    )
    fig.update_layout(
        updatemenus=[dict(
            buttons=[
                dict(args=[None, {"frame":{"duration":130,"redraw":True},
                                   "fromcurrent":True,"transition":{"duration":0}}],
                     label="▶  Play", method="animate"),
                dict(args=[[None], {"frame":{"duration":0},"mode":"immediate",
                                     "transition":{"duration":0}}],
                     label="⏸  Pause", method="animate"),
            ],
            direction="left", pad={"r":10,"t":12},
            showactive=False, type="buttons",
            x=0.08, xanchor="right", y=-0.22, yanchor="top",
            **ANIM_BUTTON_STYLE,
        )],
        sliders=[dict(
            active=0,
            bgcolor="rgba(13,25,50,0.6)",
            currentvalue=dict(font=dict(size=11,color="rgba(200,216,255,0.5)"),
                              prefix="Step ", visible=True, xanchor="center"),
            pad={"b":10,"t":55,"l":10},
            len=0.9, x=0.05, y=-0.19,
            steps=[dict(
                args=[[f.name], {"frame":{"duration":130,"redraw":True},
                                  "mode":"immediate","transition":{"duration":0}}],
                label="", method="animate",
            ) for f in frames],
        )],
    )
    return fig


def build_tt_profile(NT, QT, TT):
    """Plotly temperature-time schematic of the heat treatment cycle."""
    # Schematic (not real-time) waypoints
    t = [0,  0.5,  2,    3,    3.5,   4,    4.5,  6,    7,    7.5,   8]
    T = [25,  25,  NT,   NT,   NT//2, 25,   25,   TT,   TT,   TT//2, 25]

    a3_c = a3_temp(0.40)  # reference A3 for a typical medium-carbon steel

    fig = go.Figure()

    # ── Shaded phase regions behind the curve ──────────────────────────────
    fig.add_hrect(y0=0,   y1=A1,          fillcolor="rgba(70,170,220,0.07)",
                  line_width=0, annotation_text="Below A₁ — Ferrite/Pearlite",
                  annotation_position="bottom right",
                  annotation_font=dict(size=9, color="rgba(200,216,255,0.35)"))
    fig.add_hrect(y0=A1,  y1=a3_c,        fillcolor="rgba(255,165,60,0.07)",
                  line_width=0, annotation_text="γ + α  two-phase",
                  annotation_position="top right",
                  annotation_font=dict(size=9, color="rgba(255,165,60,0.4)"))
    fig.add_hrect(y0=a3_c, y1=max(NT,QT)*1.1, fillcolor="rgba(255,210,60,0.07)",
                  line_width=0, annotation_text="Austenite (γ)",
                  annotation_position="top right",
                  annotation_font=dict(size=9, color="rgba(255,210,60,0.4)"))

    # ── Horizontal reference lines ─────────────────────────────────────────
    for ref_T, lbl, col in [(A1, f"A₁ = {A1}°C", "#e74c3c"),
                             (a3_c, f"A₃ ≈ {a3_c:.0f}°C", "#4a90d9")]:
        fig.add_hline(y=ref_T, line_dash="dot", line_color=col,
                      line_width=1.3, opacity=0.65,
                      annotation_text=f"  {lbl}",
                      annotation_font=dict(size=9, color=col),
                      annotation_position="left")

    # ── Area fill under curve ──────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=t, y=T, fill="tozeroy",
        fillcolor="rgba(76,114,176,0.12)",
        line=dict(color="rgba(0,0,0,0)", width=0),
        showlegend=False, hoverinfo="skip",
    ))

    # ── Main temperature line ──────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=t, y=T, mode="lines+markers",
        line=dict(color="#4c72b0", width=3.5, shape="spline", smoothing=0.5),
        marker=dict(size=9, color="white", line=dict(color="#4c72b0", width=2.5)),
        name="Temperature profile",
    ))

    # ── Stage labels ──────────────────────────────────────────────────────
    stage_labels = [
        (1.25, NT+35, f"Normalising<br><b>{NT}°C</b>", "#7ec8e3"),
        (3.75, max(NT,QT)/2-30, "Quench", "#dd8452"),
        (6.5,  TT+35, f"Tempering<br><b>{TT}°C</b>",  "#f39c12"),
    ]
    for tx, ty, text, col in stage_labels:
        fig.add_annotation(x=tx, y=ty, text=text, showarrow=False,
                           font=dict(size=10, color=col, family="Rajdhani, sans-serif"),
                           align="center", bgcolor="rgba(8,13,30,0.6)",
                           bordercolor=col+"55", borderwidth=1, borderpad=5)

    fig.update_layout(**_PLOTLY_BASE)
    fig.update_layout(
        xaxis=dict(title="Process Stage (schematic time)", showticklabels=False,
                   range=[-0.2, 8.8], showgrid=False, zeroline=False, color="#c8d8ff"),
        yaxis=dict(title="Temperature (°C)", range=[0, max(NT,QT)*1.18],
                   gridcolor="rgba(255,255,255,0.05)", showgrid=True, color="#c8d8ff"),
        title=dict(text="Heat Treatment Cycle  ·  Schematic Temperature Profile",
                   font=dict(size=14, color="white", family="Rajdhani, sans-serif")),
        showlegend=False,
        height=360,
        margin=dict(l=70, r=20, t=55, b=40),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:12px 0 8px;">
      <div style="font-family:'Rajdhani',sans-serif;font-size:1.6rem;font-weight:700;
                  background:linear-gradient(135deg,#4c72b0,#00b4d8);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        ⚙ SteelSight
      </div>
      <div style="font-size:0.70rem;color:rgba(180,200,255,0.40);
                  letter-spacing:0.15em;text-transform:uppercase;margin-top:2px;">
        Mechanical Property Predictor
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Section 1: Chemical Composition ──────────────────────────────────
    st.markdown(html_section_header("Chemical Composition",
                "Element weight percentages (wt%)", "🧪"),
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        C  = st.number_input("C  (Carbon)",      0.0, 2.0,  0.400, 0.001, "%.3f")
        Mn = st.number_input("Mn  (Manganese)",  0.0, 3.0,  0.640, 0.001, "%.3f")
        Ni = st.number_input("Ni  (Nickel)",     0.0, 5.0,  0.040, 0.001, "%.3f")
        Cu = st.number_input("Cu  (Copper)",     0.0, 3.0,  0.160, 0.001, "%.3f")
        P  = st.number_input("P  (Phosphorus)",  0.0, 0.1,  0.020, 0.001, "%.3f")
    with col_b:
        Si = st.number_input("Si  (Silicon)",    0.0, 3.0,  0.970, 0.001, "%.3f")
        Cr = st.number_input("Cr  (Chromium)",   0.0, 5.0,  0.010, 0.001, "%.3f")
        Mo = st.number_input("Mo  (Molybdenum)", 0.0, 3.0,  0.000, 0.001, "%.3f")
        S  = st.number_input("S   (Sulfur)",     0.0, 0.3,  0.013, 0.001, "%.3f")

    # Steel type badge
    c_type = ("Hypo-eutectoid (heat-treatable)" if C < C_EUTECTOID
              else "Hyper-eutectoid" if C <= C_AUS_MAX else "Cast Iron range")
    c_col  = "#50c878" if C < C_EUTECTOID else "#ff8c30" if C <= C_AUS_MAX else "#e74c3c"
    st.markdown(html_badge(c_type, c_col), unsafe_allow_html=True)

    st.divider()

    # ── Section 2: Heat Treatment ─────────────────────────────────────────
    st.markdown(html_section_header("Heat Treatment",
                "All three steps in sequence: N → Q → T", "🔥"),
                unsafe_allow_html=True)

    NT = st.slider("Normalising  (°C)", 800, 950, 865, 5)
    QT = st.slider("Quenching    (°C)", 800, 950, 865, 5)
    TT = st.slider("Tempering    (°C)", 450, 750, 550, 10)

    # Real-time heat treatment validation
    a3_c = a3_temp(C)
    if NT > a3_c:
        st.success(f"✓ NT ({NT}°C) > A₃ ({a3_c:.0f}°C) — Fully austenitic", icon="✅")
    else:
        st.warning(f"⚠ NT ({NT}°C) < A₃ ({a3_c:.0f}°C) — Incomplete austenitisation")
    if TT < A1:
        st.success(f"✓ TT ({TT}°C) < A₁ ({A1}°C) — Correct sub-critical temper", icon="✅")
    else:
        st.error(f"✗ TT ({TT}°C) ≥ A₁ ({A1}°C) — Over-tempering risk")

    st.divider()

    predict_clicked = st.button("🔮  Predict Mechanical Properties",
                                 type="primary", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HERO BANNER
# ══════════════════════════════════════════════════════════════════════════════
_hero_html = """
<div style="
    background: linear-gradient(135deg, #0d1f3c 0%, #112240 40%, #0d2137 70%, #08111e 100%);
    border: 1px solid rgba(76,114,176,0.30);
    border-radius: 18px;
    padding: 32px 40px 28px;
    margin-bottom: 22px;
    position: relative;
    overflow: hidden;
">
  <div style="position:absolute;top:-60px;right:-40px;width:300px;height:300px;
              border-radius:50%;background:radial-gradient(circle,rgba(76,114,176,0.12),transparent 70%);
              pointer-events:none;"></div>
  <div style="position:absolute;bottom:-80px;left:30%;width:400px;height:200px;
              border-radius:50%;background:radial-gradient(circle,rgba(0,180,216,0.06),transparent 70%);
              pointer-events:none;"></div>

  <div style="display:flex;align-items:center;gap:16px;margin-bottom:10px;">
    <span style="font-size:2.4rem;">⚙️</span>
    <div>
      <div style="font-family:'Rajdhani',sans-serif;font-size:2rem;font-weight:700;
                  color:white;line-height:1;letter-spacing:0.02em;">
        Steel Heat Treatment
        <span style="background:linear-gradient(135deg,#4c72b0,#00b4d8);
                     -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        Property Predictor
        </span>
      </div>
      <div style="color:rgba(180,200,255,0.55);font-size:0.88rem;margin-top:6px;">
        XGBoost multi-output regression  ·  NIMS fatigue database  ·  361 heat-treated steel samples
      </div>
    </div>
  </div>

  <div style="display:flex;flex-wrap:wrap;gap:0;margin-top:14px;">
"""
# Dynamic stats from model metrics
if MODEL_METRICS:
    avg_r2 = sum(m["R²"] for m in MODEL_METRICS["metrics"]) / len(MODEL_METRICS["metrics"])
    n_train = MODEL_METRICS.get("n_train", 289)
    r2_str  = f"{avg_r2*100:.2f}%"
    _hero_html += html_stat_pill("Model", "XGBoost v2 (per-target)", "#4c72b0")
    _hero_html += html_stat_pill("Avg R²", r2_str, "#50c878")
    _hero_html += html_stat_pill("Training samples", str(n_train), "#dd8452")
    for m in MODEL_METRICS["metrics"]:
        _hero_html += html_stat_pill(f'{m["Target"]} R²', f'{m["R²"]*100:.1f}%', "#00b4d8")
else:
    _hero_html += html_stat_pill("Model", "XGBoost", "#4c72b0")
    _hero_html += html_stat_pill("R² Score", "97.35%", "#50c878")
    _hero_html += html_stat_pill("Training samples", "289", "#dd8452")
    _hero_html += html_stat_pill("Outputs", "Tensile · Hardness · Fatigue", "#00b4d8")

_hero_html += """
  </div>
</div>
"""
st.html(_hero_html)

# ══════════════════════════════════════════════════════════════════════════════
#  RUN PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
result = None
if predict_clicked:
    inp = dict(C=C,Si=Si,Mn=Mn,P=P,S=S,Ni=Ni,Cr=Cr,Cu=Cu,Mo=Mo,NT=NT,TT=TT,QT=QT)
    ood = check_ood({k:inp[k] for k in TRAINING_RANGES})
    tensile, hardness, fatigue = predict_properties(inp)
    result   = (tensile, hardness, fatigue, ood)
    st.session_state["result"] = result
elif "result" in st.session_state:
    result = st.session_state["result"]

# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_res, tab_phase, tab_anim, tab_sens, tab_batch, tab_data = st.tabs([
    "📊  Prediction Results",
    "🔬  Phase Diagram",
    "🎬  Process Animation",
    "📈  Sensitivity Analysis",
    "📁  Batch Predictions",
    "🗃️  Dataset Explorer",
])

# ────────────────────────────────────────────────────────────────────────────
#  TAB 1  ─  RESULTS
# ────────────────────────────────────────────────────────────────────────────
with tab_res:
    if result is None:
        st.markdown("""
        <div style="
            text-align:center;padding:60px 40px;
            background:rgba(13,25,50,0.5);border:1px dashed rgba(76,114,176,0.25);
            border-radius:16px;margin:20px 0;
        ">
          <div style="font-size:3rem;margin-bottom:16px;">🔮</div>
          <div style="font-family:'Rajdhani',sans-serif;font-size:1.5rem;
                      color:#c8d8ff;font-weight:600;margin-bottom:8px;">
            Ready to Predict
          </div>
          <div style="color:rgba(180,200,255,0.45);font-size:0.88rem;max-width:440px;margin:0 auto;">
            Set your steel's chemical composition and heat treatment temperatures
            in the sidebar, then click <b style="color:#4c72b0;">Predict Mechanical Properties</b>.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        tensile, hardness, fatigue, ood = result
        if ood:
            st.warning("**Out-of-distribution inputs:**\n\n" + "  ·  ".join(ood) +
                       "\n\nPredictions may be less reliable.")

        grade_name, grade_col = steel_grade(tensile)

        # ── Header row ──────────────────────────────────────────────────
        hcol1, hcol2 = st.columns([3, 1])
        with hcol1:
            st.markdown(html_section_header("Predicted Properties",
                        "Based on XGBoost model trained on NIMS fatigue database", "📊"),
                        unsafe_allow_html=True)
        with hcol2:
            st.markdown(
                f'<div style="text-align:right;padding-top:8px;">'
                f'{html_badge(grade_name, grade_col)}</div>',
                unsafe_allow_html=True)

        # ── Metric cards ─────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(html_metric_card(
                "Tensile Strength", f"{tensile:.1f}", "MPa",
                "#4c72b0", "💪",
                f"Range: {PROP_RANGES['Tensile'][0]}–{PROP_RANGES['Tensile'][1]} MPa"
            ), unsafe_allow_html=True)
        with c2:
            st.markdown(html_metric_card(
                "Hardness", f"{hardness:.1f}", "HB (Brinell)",
                "#dd8452", "🪨",
                f"Range: {PROP_RANGES['Hardness'][0]}–{PROP_RANGES['Hardness'][1]} HB"
            ), unsafe_allow_html=True)
        with c3:
            st.markdown(html_metric_card(
                "Fatigue Strength", f"{fatigue:.1f}", "MPa",
                "#55a868", "🔄",
                f"Range: {PROP_RANGES['Fatigue'][0]}–{PROP_RANGES['Fatigue'][1]} MPa"
            ), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Gauges + Radar ──────────────────────────────────────────────
        st.plotly_chart(build_gauge_charts(tensile, hardness, fatigue),
                        use_container_width=True)

        gc1, gc2 = st.columns([3, 2])
        with gc2:
            st.markdown(html_section_header("Property Profile",
                        "Normalised vs. dataset range", "🎯"),
                        unsafe_allow_html=True)
            st.plotly_chart(build_radar_chart(tensile, hardness, fatigue),
                            use_container_width=True)
        with gc1:
            # ── Phase table ──────────────────────────────────────────────
            st.markdown(html_section_header("Equilibrium Phase at Each Step",
                        "Based on Fe-C phase diagram", "🔬"),
                        unsafe_allow_html=True)
            phase_data = [
                ("Normalising",     NT, get_phase(C, NT),  f"A₃ ≈ {a3_temp(C):.0f}°C"),
                ("Quenching",       QT, get_phase(C, QT),  "Rapid cooling → martensite"),
                ("Tempering",       TT, get_phase(C, TT),  "Below A₁ — sub-critical"),
                ("Room Temperature",25, get_phase(C, 25),  "Final microstructure"),
            ]
            for step, temp, phase, note in phase_data:
                st.markdown(f"""
                <div style="
                    display:flex;align-items:center;justify-content:space-between;
                    background:rgba(13,25,50,0.55);
                    border:1px solid rgba(76,114,176,0.18);
                    border-radius:10px;padding:12px 16px;margin-bottom:8px;
                ">
                  <div>
                    <div style="font-weight:600;color:#e8f0ff;font-size:0.88rem;">{step}</div>
                    <div style="color:rgba(180,200,255,0.45);font-size:0.75rem;margin-top:2px;">{note}</div>
                  </div>
                  <div style="text-align:right;">
                    <div style="font-family:'JetBrains Mono',monospace;color:#4c72b0;
                                font-weight:600;font-size:1rem;">{temp}°C</div>
                    <div style="margin-top:4px;">{html_phase_badge(phase)}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        # ── Input summary ────────────────────────────────────────────────
        with st.expander("📋  View input summary"):
            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown("**Chemical Composition**")
                comp_df = pd.DataFrame({
                    "Element": ["C","Si","Mn","P","S","Ni","Cr","Cu","Mo"],
                    "wt%":     [C, Si, Mn, P, S, Ni, Cr, Cu, Mo],
                })
                st.dataframe(comp_df.style.format({"wt%":"{:.3f}"}),
                             hide_index=True, use_container_width=True)
            with sc2:
                st.markdown("**Heat Treatment**")
                ht_df = pd.DataFrame({
                    "Step":        ["Normalising","Quenching","Tempering"],
                    "Temp (°C)":   [NT, QT, TT],
                    "In Range?":   [
                        "✅" if 825<=NT<=900 else "⚠️",
                        "✅" if 825<=QT<=870 else "⚠️",
                        "✅" if 550<=TT<=680 else "⚠️",
                    ],
                })
                st.dataframe(ht_df, hide_index=True, use_container_width=True)

        # ── Export prediction ────────────────────────────────────────────
        export_row = {
            "C": C, "Si": Si, "Mn": Mn, "P": P, "S": S,
            "Ni": Ni, "Cr": Cr, "Cu": Cu, "Mo": Mo,
            "NT": NT, "QT": QT, "TT": TT,
            "Tensile (MPa)": round(tensile, 1),
            "Hardness (HB)": round(hardness, 1),
            "Fatigue (MPa)": round(fatigue, 1),
        }
        csv_single = pd.DataFrame([export_row]).to_csv(index=False)
        st.download_button("⬇️  Download Prediction CSV", csv_single,
                           "steelsight_prediction.csv", "text/csv",
                           use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
#  TAB 2  ─  PHASE DIAGRAM
# ────────────────────────────────────────────────────────────────────────────
with tab_phase:
    st.markdown(html_section_header("Fe-C Phase Diagram",
                "Interactive — hover, zoom, pan · Steel region 0–2.3 wt% C", "🔬"),
                unsafe_allow_html=True)
    st.plotly_chart(build_phase_diagram(C, NT, QT, TT, animated=False),
                    use_container_width=True)

    with st.expander("ℹ️  How to read this diagram"):
        rc1, rc2 = st.columns(2)
        with rc1:
            st.markdown("""
**Phase Boundaries**
| Line | Meaning |
|------|---------|
| **A₁** (red, 723°C) | Eutectoid temperature — below this line steel has no austenite |
| **A₃** (blue, sloping) | Upper critical temp — above this, steel is *fully* austenitic |
| **Acm** (green, sloping) | Austenite/cementite boundary for high-carbon steels |
| **Solidus** (gray dashed) | Boundary above which liquid phase appears |
            """)
        with rc2:
            st.markdown("""
**Heat Treatment Logic**
| Condition | Status |
|-----------|--------|
| NT, QT **above A₃** | ✅ Fully austenitic before quench |
| NT, QT **between A₁–A₃** | ⚠️ Two-phase — incomplete treatment |
| TT **below A₁** | ✅ Correct sub-critical tempering |
| TT **above A₁** | ✗ Steel re-austenitises — properties lost |
            """)

# ────────────────────────────────────────────────────────────────────────────
#  TAB 3  ─  PROCESS ANIMATION
# ────────────────────────────────────────────────────────────────────────────
with tab_anim:
    st.markdown(html_section_header("Heat Treatment Process Animation",
                "Watch the steel traverse the Fe-C diagram · Press ▶ Play below the chart", "🎬"),
                unsafe_allow_html=True)

    st.plotly_chart(build_phase_diagram(C, NT, QT, TT, animated=True),
                    use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(html_section_header("Temperature-Time Profile",
                "Schematic heat treatment cycle", "📈"),
                unsafe_allow_html=True)
    st.plotly_chart(build_tt_profile(NT, QT, TT), use_container_width=True)

    with st.expander("📖  Metallurgical explanation of each stage"):
        st.markdown(f"""
**Stage 1 — Normalising at {NT}°C**
- Steel heated above A₃ ≈ {a3_temp(C):.0f}°C (for C = {C:.3f}%)
- Transforms entirely to **austenite (γ)** — face-centred cubic (FCC), dissolves up to 2.11% C
- Carbides dissolve; grain structure homogenises and grain size is refined

**Stage 2 — Quenching (rapid cooling)**
- Cooling is too fast for diffusion → austenite cannot decompose to ferrite+pearlite
- Instead it shears into **martensite** — a supersaturated body-centred tetragonal (BCT) phase
- Martensite is extremely hard but brittle; it is *not* on the equilibrium diagram (note the dot jumps to the ferrite+pearlite region — that is the equilibrium prediction, not the actual microstructure)

**Stage 3 — Tempering at {TT}°C**
- Reheating below A₁ (723°C) — no phase change by equilibrium, but kinetics matter
- Carbon atoms diffuse out of the martensite lattice, precipitating as fine carbide particles
- Result: **tempered martensite** — significantly better toughness/ductility while retaining most hardness

**Final microstructure**
- Tempered martensite with fine carbide dispersion
- Mechanical properties: Tensile ≈ predicted value, Hardness ≈ predicted value, Fatigue life ≈ predicted value
        """)

# ────────────────────────────────────────────────────────────────────────────
#  TAB 4  ─  SENSITIVITY ANALYSIS
# ────────────────────────────────────────────────────────────────────────────
with tab_sens:
    st.markdown(html_section_header("Sensitivity Analysis",
                "Explore how each input affects predicted properties", "📈"),
                unsafe_allow_html=True)

    if result is None:
        st.info("🔮  Run a prediction first — sensitivity analysis uses the current sidebar inputs as the baseline.")
    else:
        base_inp = dict(C=C, Si=Si, Mn=Mn, P=P, S=S, Ni=Ni, Cr=Cr, Cu=Cu, Mo=Mo,
                        NT=NT, TT=TT, QT=QT)
        base_t, base_h, base_f = result[0], result[1], result[2]

        # ── One-at-a-time sensitivity ───────────────────────────────────
        sens_feature = st.selectbox(
            "Select a feature to vary",
            BASE_FEATURES,
            index=0,
            help="All other features stay at their current sidebar values."
        )

        feat_val = base_inp[sens_feature]
        is_temp = sens_feature in ("NT", "QT", "TT")

        if is_temp:
            low  = max(400, feat_val - 200)
            high = min(1000, feat_val + 200)
            sweep = np.linspace(low, high, 50)
        else:
            low  = max(0.0, feat_val * 0.2)
            high = feat_val * 3 if feat_val > 0 else 0.5
            sweep = np.linspace(low, high, 50)

        sweep_results = {"x": [], "Tensile": [], "Hardness": [], "Fatigue": []}
        for v in sweep:
            test_inp = base_inp.copy()
            test_inp[sens_feature] = float(v)
            t, h, f = predict_properties(test_inp)
            sweep_results["x"].append(v)
            sweep_results["Tensile"].append(t)
            sweep_results["Hardness"].append(h)
            sweep_results["Fatigue"].append(f)

        unit = "°C" if is_temp else "wt%"

        # ── Line chart ─────────────────────────────────────────────────
        fig_sens = make_subplots(rows=1, cols=3,
                                 subplot_titles=["Tensile Strength (MPa)",
                                                 "Hardness (HB)",
                                                 "Fatigue Strength (MPa)"])

        for col_i, (prop, colour) in enumerate([
            ("Tensile", "#4c72b0"), ("Hardness", "#dd8452"), ("Fatigue", "#55a868")
        ], 1):
            fig_sens.add_trace(go.Scatter(
                x=sweep_results["x"], y=sweep_results[prop],
                mode="lines", line=dict(color=colour, width=3),
                name=prop
            ), row=1, col=col_i)
            # Mark baseline
            base_val = {"Tensile": base_t, "Hardness": base_h, "Fatigue": base_f}[prop]
            fig_sens.add_trace(go.Scatter(
                x=[feat_val], y=[base_val], mode="markers",
                marker=dict(size=12, color="white", line=dict(color=colour, width=3)),
                name=f"Current ({feat_val})", showlegend=(col_i == 1),
            ), row=1, col=col_i)

        fig_sens.update_layout(**_PLOTLY_BASE)
        fig_sens.update_layout(
            height=380,
            title=dict(text=f"Effect of {sens_feature} on Mechanical Properties",
                       font=dict(size=14, color="white", family="Rajdhani, sans-serif")),
            showlegend=True,
            legend=dict(bgcolor="rgba(8,13,30,0.7)", bordercolor="rgba(76,114,176,0.3)",
                        borderwidth=1, font=dict(size=10)),
        )
        fig_sens.update_xaxes(title_text=f"{sens_feature} ({unit})")
        st.plotly_chart(fig_sens, use_container_width=True)

        # ── Tornado chart: all features at ±10% ────────────────────────
        st.markdown(html_section_header("Impact Ranking",
                    "Change in Tensile Strength when each feature varies ±10% from current value", "🌪️"),
                    unsafe_allow_html=True)

        tornado_data = []
        for feat in BASE_FEATURES:
            test_lo = base_inp.copy()
            test_hi = base_inp.copy()
            val = base_inp[feat]
            if feat in ("NT", "QT", "TT"):
                delta = 20
            else:
                delta = max(val * 0.10, 0.005)
            test_lo[feat] = val - delta
            test_hi[feat] = val + delta
            t_lo, _, _ = predict_properties(test_lo)
            t_hi, _, _ = predict_properties(test_hi)
            tornado_data.append((feat, t_lo - base_t, t_hi - base_t))

        tornado_data.sort(key=lambda x: abs(x[2] - x[1]))

        fig_tornado = go.Figure()
        for feat, lo, hi in tornado_data:
            fig_tornado.add_trace(go.Bar(
                y=[feat], x=[hi - lo], orientation="h",
                marker_color="#4c72b0" if hi > lo else "#dd8452",
                name=feat, showlegend=False,
                hovertemplate=f"{feat}<br>Low: {lo:+.1f} MPa<br>High: {hi:+.1f} MPa<extra></extra>",
            ))

        fig_tornado.update_layout(**_PLOTLY_BASE)
        fig_tornado.update_layout(
            height=350, xaxis_title="Δ Tensile Strength (MPa)",
            title=dict(text="Feature Impact — Tornado Chart",
                       font=dict(size=14, color="white", family="Rajdhani, sans-serif")),
        )
        st.plotly_chart(fig_tornado, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
#  TAB 5  ─  BATCH PREDICTIONS
# ────────────────────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown(html_section_header("Batch Predictions",
                "Upload a CSV file to predict properties for multiple steel compositions", "📁"),
                unsafe_allow_html=True)

    st.markdown("""
    <div style="
        background:rgba(13,25,50,0.55);border:1px solid rgba(76,114,176,0.25);
        border-radius:12px;padding:16px 20px;margin-bottom:16px;
    ">
      <div style="color:#c8d8ff;font-size:0.88rem;font-weight:600;margin-bottom:8px;">
        📋 Required CSV columns
      </div>
      <div style="color:rgba(180,200,255,0.55);font-size:0.82rem;font-family:'JetBrains Mono',monospace;">
        C, Si, Mn, P, S, Ni, Cr, Cu, Mo, NT, TT, QT
      </div>
      <div style="color:rgba(180,200,255,0.35);font-size:0.75rem;margin-top:6px;">
        Each row = one steel composition. Temperature columns in °C, elements in wt%.
      </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            missing = [c for c in BASE_FEATURES if c not in batch_df.columns]
            if missing:
                st.error(f"❌ Missing columns: {', '.join(missing)}")
            else:
                st.success(f"✅ Loaded {len(batch_df)} rows")

                # Predict
                batch_results = []
                progress = st.progress(0, text="Predicting…")
                for idx, row in batch_df.iterrows():
                    inp = {f: float(row[f]) for f in BASE_FEATURES}
                    t, h, f_ = predict_properties(inp)
                    batch_results.append({"Row": idx+1, **inp,
                                          "Tensile (MPa)": round(t, 1),
                                          "Hardness (HB)": round(h, 1),
                                          "Fatigue (MPa)": round(f_, 1)})
                    progress.progress((idx+1) / len(batch_df))
                progress.empty()

                results_df = pd.DataFrame(batch_results)

                # Summary stats
                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    st.markdown(html_metric_card(
                        "Avg Tensile", f"{results_df['Tensile (MPa)'].mean():.0f}",
                        "MPa", "#4c72b0", "💪"), unsafe_allow_html=True)
                with sc2:
                    st.markdown(html_metric_card(
                        "Avg Hardness", f"{results_df['Hardness (HB)'].mean():.0f}",
                        "HB", "#dd8452", "🪨"), unsafe_allow_html=True)
                with sc3:
                    st.markdown(html_metric_card(
                        "Avg Fatigue", f"{results_df['Fatigue (MPa)'].mean():.0f}",
                        "MPa", "#55a868", "🔄"), unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Results table
                st.dataframe(results_df, use_container_width=True, hide_index=True)

                # Distribution plots
                fig_batch = make_subplots(rows=1, cols=3,
                    subplot_titles=["Tensile (MPa)", "Hardness (HB)", "Fatigue (MPa)"])
                for col_i, (col, colour) in enumerate([
                    ("Tensile (MPa)", "#4c72b0"),
                    ("Hardness (HB)", "#dd8452"),
                    ("Fatigue (MPa)", "#55a868"),
                ], 1):
                    fig_batch.add_trace(go.Histogram(
                        x=results_df[col], marker_color=colour,
                        opacity=0.8, name=col
                    ), row=1, col=col_i)
                fig_batch.update_layout(**_PLOTLY_BASE)
                fig_batch.update_layout(height=300, showlegend=False,
                    title=dict(text="Batch Prediction Distributions",
                               font=dict(size=14, color="white", family="Rajdhani")))
                st.plotly_chart(fig_batch, use_container_width=True)

                # Download
                csv_out = results_df.to_csv(index=False)
                st.download_button("⬇️  Download Results CSV", csv_out,
                                   "steelsight_batch_results.csv", "text/csv",
                                   use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

    else:
        # Download template
        template = pd.DataFrame([{f: "" for f in BASE_FEATURES}])
        st.download_button("📥  Download CSV Template", template.to_csv(index=False),
                           "steelsight_template.csv", "text/csv")


# ────────────────────────────────────────────────────────────────────────────
#  TAB 6  ─  DATASET EXPLORER
# ────────────────────────────────────────────────────────────────────────────
with tab_data:
    st.markdown(html_section_header("NIMS Dataset Explorer",
                "Browse the training dataset and find similar steel compositions", "🗃️"),
                unsafe_allow_html=True)

    @st.cache_data
    def load_dataset():
        csv_path = os.path.join(os.path.dirname(__file__), "NIMS_Fatigue.csv")
        if not os.path.exists(csv_path):
            csv_path = "NIMS_Fatigue.csv"
        return pd.read_csv(csv_path)

    try:
        dataset = load_dataset()

        dc1, dc2 = st.columns([1, 1])
        with dc1:
            x_feat = st.selectbox("X-axis", BASE_FEATURES + TARGETS, index=0, key="dx")
        with dc2:
            y_feat = st.selectbox("Y-axis", BASE_FEATURES + TARGETS, index=len(BASE_FEATURES), key="dy")

        fig_data = go.Figure()
        fig_data.add_trace(go.Scatter(
            x=dataset[x_feat], y=dataset[y_feat],
            mode="markers",
            marker=dict(size=7, color=dataset.get("Tensile", dataset[y_feat]),
                        colorscale="Viridis", showscale=True,
                        colorbar=dict(title="Tensile")),
            text=[f"Row {i}" for i in range(len(dataset))],
            hovertemplate=f"{x_feat}: %{{x}}<br>{y_feat}: %{{y}}<extra></extra>",
        ))

        # Mark current input if prediction exists
        if result is not None and x_feat in BASE_FEATURES and y_feat in TARGETS:
            curr = {"C": C, "Si": Si, "Mn": Mn, "P": P, "S": S,
                    "Ni": Ni, "Cr": Cr, "Cu": Cu, "Mo": Mo,
                    "NT": NT, "QT": QT, "TT": TT,
                    "Tensile": result[0], "Hardness": result[1], "Fatigue": result[2]}
            if x_feat in curr and y_feat in curr:
                fig_data.add_trace(go.Scatter(
                    x=[curr[x_feat]], y=[curr[y_feat]],
                    mode="markers+text", text=["  Your Steel"],
                    textposition="top right",
                    textfont=dict(size=12, color="white"),
                    marker=dict(size=16, color="red", symbol="star",
                                line=dict(color="white", width=2)),
                    name="Your prediction",
                ))

        fig_data.update_layout(**_PLOTLY_BASE)
        fig_data.update_layout(
            xaxis_title=x_feat, yaxis_title=y_feat,
            height=450,
            title=dict(text=f"{x_feat} vs {y_feat} — NIMS Fatigue Database ({len(dataset)} samples)",
                       font=dict(size=14, color="white", family="Rajdhani")),
        )
        st.plotly_chart(fig_data, use_container_width=True)

        # ── Dataset stats ──────────────────────────────────────────────
        with st.expander("📊 Dataset Statistics"):
            st.dataframe(dataset[BASE_FEATURES + TARGETS].describe().round(3),
                         use_container_width=True)

        # ── Nearest neighbour ──────────────────────────────────────────
        if result is not None:
            st.markdown(html_section_header("Nearest Neighbours",
                        "Training samples most similar to your input", "🔍"),
                        unsafe_allow_html=True)

            curr_vals = np.array([C, Si, Mn, P, S, Ni, Cr, Cu, Mo, NT, TT, QT])
            ds_vals   = dataset[BASE_FEATURES].values
            # Normalise for distance
            std = ds_vals.std(axis=0)
            std[std == 0] = 1
            dists = np.sqrt(((ds_vals - curr_vals) / std) ** 2).sum(axis=1)
            top_idx = dists.argsort()[:5]

            nn_df = dataset.iloc[top_idx][BASE_FEATURES + TARGETS].copy()
            nn_df.insert(0, "Distance", [f"{dists[i]:.2f}" for i in top_idx])
            st.dataframe(nn_df, use_container_width=True, hide_index=True)

        # ── Full table ─────────────────────────────────────────────────
        with st.expander("📋 Full Dataset"):
            st.dataframe(dataset, use_container_width=True, hide_index=True)

    except FileNotFoundError:
        st.warning("⚠️ NIMS_Fatigue.csv not found. Place it in the project directory.")


# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
_model_badge = f"v2 per-target" if MODEL_MODE == "v2" else "v1 multi-output"
st.markdown(f"""
<div style="
    text-align:center;padding:20px 0 10px;margin-top:40px;
    border-top:1px solid rgba(76,114,176,0.15);
">
  <div style="font-size:0.72rem;color:rgba(180,200,255,0.30);letter-spacing:0.1em;">
    SteelSight · Model: XGBoost {_model_badge} · Dataset: NIMS Fatigue ({len(load_dataset()) if 'load_dataset' in dir() else 362} samples)
    · Built with Streamlit + Plotly
  </div>
</div>
""", unsafe_allow_html=True)
