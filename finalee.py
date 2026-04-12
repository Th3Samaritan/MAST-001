"""
SteelSight v2 — Heat Treatment Mechanical Property Predictor
=============================================================
5 targets: Tensile · Yield · Hardness · Elongation · Fatigue
4 process types with conditional inputs
"""

import os, json as json_lib
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
    page_title="SteelSight — Heat Treatment Predictor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden; }

.stApp {
    background: radial-gradient(ellipse at 20% 0%, #0d1f3c 0%, #06090f 50%, #0a0d1a 100%);
    min-height: 100vh;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#080d1e 0%,#0b1224 60%,#06090f 100%) !important;
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
.stNumberInput input {
    background: rgba(13,25,50,0.85) !important;
    border: 1px solid rgba(76,114,176,0.30) !important;
    border-radius: 8px !important;
    color: #e8f0ff !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.88rem !important;
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
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg,#4c72b0,#00b4d8) !important;
}
[data-testid="stSlider"] [role="slider"] {
    background: white !important;
    border: 2px solid #4c72b0 !important;
    box-shadow: 0 0 8px rgba(76,114,176,0.6) !important;
}
[data-testid="stSlider"] p { font-family:'JetBrains Mono',monospace !important; font-size:0.8rem !important; color:#7b9fd4 !important; }
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#1a3a6b 0%,#2557ab 50%,#1e4d9e 100%) !important;
    border: 1px solid rgba(76,114,176,0.55) !important;
    color: #e8f4ff !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.65rem 1.5rem !important;
    box-shadow: 0 4px 20px rgba(37,87,171,0.45) !important;
    transition: all 0.25s ease !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg,#2557ab 0%,#3473c7 50%,#2a63b5 100%) !important;
    box-shadow: 0 6px 30px rgba(52,115,199,0.60) !important;
    transform: translateY(-2px) !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: rgba(8,13,30,0.8) !important;
    border: 1px solid rgba(76,114,176,0.2) !important;
    border-radius: 12px !important;
    padding: 5px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: rgba(180,200,240,0.55) !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 0.45rem 1.2rem !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg,rgba(76,114,176,0.30),rgba(0,180,216,0.15)) !important;
    color: #e8f0ff !important;
}
[data-testid="stExpander"] {
    background: rgba(8,13,30,0.6) !important;
    border: 1px solid rgba(76,114,176,0.18) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary { color: rgba(180,200,240,0.8) !important; font-size:0.85rem !important; }
hr { border-color: rgba(76,114,176,0.20) !important; }
.block-container { padding: 1.8rem 2.5rem 1rem !important; }
.stSelectbox > div > div {
    background: rgba(13,25,50,0.85) !important;
    border: 1px solid rgba(76,114,176,0.35) !important;
    border-radius: 8px !important;
    color: #e8f0ff !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HTML HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def html_metric_card(title, value, unit, color, icon, subtitle=""):
    return f"""
    <div style="
        background:linear-gradient(145deg,rgba(13,25,50,0.95),rgba(8,13,30,0.98));
        border:1px solid {color}55; border-top:3px solid {color};
        border-radius:14px; padding:20px 16px 16px;
        text-align:center;
        box-shadow:0 8px 32px rgba(0,0,0,0.45),0 0 30px {color}18;
        position:relative; overflow:hidden;
    ">
      <div style="position:absolute;top:-30px;right:-20px;font-size:6rem;opacity:0.04;">{icon}</div>
      <div style="font-size:2.0rem;margin-bottom:5px;">{icon}</div>
      <div style="font-size:2.4rem;font-weight:700;color:{color};
                  font-family:'Rajdhani',sans-serif;line-height:1;
                  letter-spacing:-0.02em;">{value}</div>
      <div style="font-size:0.68rem;color:rgba(180,200,255,0.45);
                  text-transform:uppercase;letter-spacing:0.14em;
                  margin-top:8px;font-weight:600;">{title}</div>
      <div style="font-size:0.76rem;color:{color}99;margin-top:2px;">{unit}</div>
      {f'<div style="font-size:0.70rem;color:rgba(180,200,255,0.35);margin-top:5px;">{subtitle}</div>' if subtitle else ''}
    </div>"""


def html_section_header(title, subtitle="", icon=""):
    return f"""
    <div style="margin:0 0 16px 0;">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
        <span style="font-size:1.3rem;">{icon}</span>
        <span style="font-family:'Rajdhani',sans-serif;font-size:1.35rem;
                     font-weight:700;color:#e8f0ff;letter-spacing:0.03em;">{title}</span>
      </div>
      {f'<p style="color:rgba(180,200,255,0.50);font-size:0.82rem;margin:0 0 0 2.2rem;">{subtitle}</p>' if subtitle else ''}
    </div>"""


def html_badge(text, color):
    return (f'<span style="display:inline-block;padding:3px 10px;border-radius:20px;'
            f'background:{color}33;border:1px solid {color}66;color:{color};'
            f'font-size:0.72rem;font-weight:600;letter-spacing:0.06em;'
            f'text-transform:uppercase;">{text}</span>')


def html_process_card(proc, color, icon, desc):
    return f"""
    <div style="background:rgba(13,25,50,0.7);border:1px solid {color}44;
                border-left:3px solid {color};border-radius:10px;
                padding:12px 14px;margin-bottom:6px;">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
        <span style="font-size:1.1rem;">{icon}</span>
        <span style="font-family:'Rajdhani',sans-serif;font-size:1.05rem;
                     font-weight:700;color:{color};">{proc}</span>
      </div>
      <div style="font-size:0.75rem;color:rgba(180,200,255,0.55);">{desc}</div>
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
#  PHYSICAL METALLURGY HELPERS  (used for real-time feedback & feature eng.)
# ══════════════════════════════════════════════════════════════════════════════

def calc_a3(C, Mn=0.0, Si=0.0, Ni=0.0, Cr=0.0, Mo=0.0):
    return 912.0 - 203.0*np.sqrt(max(C, 1e-6)) - 30.0*Mn + 44.7*Si - 15.2*Ni + 31.5*Mo

def calc_carbon_equiv(C, Mn, Si, Ni, Cr, Mo, Cu):
    return C + Mn/6 + Si/24 + Ni/40 + Cr/5 + Mo/4 + Cu/15

def calc_hollomon_jaffe(T_C, t_min):
    if T_C <= 0 or t_min <= 0:
        return 0.0
    t_h = max(t_min / 60.0, 0.001)
    return (T_C + 273.15) * (18.0 + np.log10(t_h))

def get_phase(c, T):
    A1 = 723; A3_FE = 912; EUTECTIC_T = 1147
    C_EUT = 0.76; C_AUS_MAX = 2.11
    def a3(cc): return A3_FE - (A3_FE - A1) / C_EUT * min(cc, C_EUT)
    def acm(cc): return A1 + (EUTECTIC_T - A1) / (C_AUS_MAX - C_EUT) * (cc - C_EUT)
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
    "Tensile_MPa":    ("Tensile Strength", "MPa",  "💪", "#4c72b0"),
    "Yield_MPa":      ("Yield Strength",   "MPa",  "🔩", "#dd8452"),
    "Hardness_HB":    ("Hardness",         "HB",   "💎", "#55a868"),
    "Elongation_pct": ("Elongation",       "%",    "📏", "#c44e52"),
    "Fatigue_MPa":    ("Fatigue Strength", "MPa",  "🔄", "#8172b3"),
}
PROP_RANGES = {
    "Tensile_MPa":    (300,  2200),
    "Yield_MPa":      (150,  1900),
    "Hardness_HB":    (80,   680),
    "Elongation_pct": (3,    48),
    "Fatigue_MPa":    (100,  1000),
}

@st.cache_resource(show_spinner="Loading model…")
def load_models():
    models_dir = "models"
    metadata   = None

    meta_path = os.path.join(models_dir, "model_metrics.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json_lib.load(f)

    # Per-target JSON models (preferred)
    per_target = {}
    for tgt in TARGETS:
        p = os.path.join(models_dir, f"xgb_{tgt.lower()}.json")
        if os.path.exists(p):
            m = XGBRegressor(); m.load_model(p)
            per_target[tgt] = m

    if len(per_target) == len(TARGETS):
        return {"mode": "per_target", "models": per_target, "metadata": metadata}

    # Fallback: legacy v1 model
    if os.path.exists("xgb_model.json"):
        m = XGBRegressor(); m.load_model("xgb_model.json")
        return {"mode": "legacy", "models": m, "metadata": None}

    st.error("⚠️ No model found. Run `FYP_finale_v2.ipynb` first to train and save the model.")
    st.stop()


model_pack = load_models()
MODEL_MODE = model_pack["mode"]
META       = model_pack["metadata"] or {}
FEATURE_COLS = META.get("features", [])
TRAINING_RANGES = {
    k: (v["min"], v["max"])
    for k, v in META.get("training_ranges", {}).items()
}

PROCESS_DUMMIES = {
    "Quench_Temper":  {"Process_Quench_Temper":1,"Process_Normalizing":0,"Process_Annealing":0,"Process_Stress_Relief":0},
    "Normalizing":    {"Process_Quench_Temper":0,"Process_Normalizing":1,"Process_Annealing":0,"Process_Stress_Relief":0},
    "Annealing":      {"Process_Quench_Temper":0,"Process_Normalizing":0,"Process_Annealing":1,"Process_Stress_Relief":0},
    "Stress_Relief":  {"Process_Quench_Temper":0,"Process_Normalizing":0,"Process_Annealing":0,"Process_Stress_Relief":1},
}
MEDIUM_DUMMIES = {
    "Water":     {"Cooling_Medium_Water":1,"Cooling_Medium_Oil":0,"Cooling_Medium_Polymer":0,"Cooling_Medium_Air":0,"Cooling_Medium_Furnace":0,"Cooling_Medium_Salt Bath":0},
    "Oil":       {"Cooling_Medium_Water":0,"Cooling_Medium_Oil":1,"Cooling_Medium_Polymer":0,"Cooling_Medium_Air":0,"Cooling_Medium_Furnace":0,"Cooling_Medium_Salt Bath":0},
    "Polymer":   {"Cooling_Medium_Water":0,"Cooling_Medium_Oil":0,"Cooling_Medium_Polymer":1,"Cooling_Medium_Air":0,"Cooling_Medium_Furnace":0,"Cooling_Medium_Salt Bath":0},
    "Air":       {"Cooling_Medium_Water":0,"Cooling_Medium_Oil":0,"Cooling_Medium_Polymer":0,"Cooling_Medium_Air":1,"Cooling_Medium_Furnace":0,"Cooling_Medium_Salt Bath":0},
    "Furnace":   {"Cooling_Medium_Water":0,"Cooling_Medium_Oil":0,"Cooling_Medium_Polymer":0,"Cooling_Medium_Air":0,"Cooling_Medium_Furnace":1,"Cooling_Medium_Salt Bath":0},
    "Salt Bath": {"Cooling_Medium_Water":0,"Cooling_Medium_Oil":0,"Cooling_Medium_Polymer":0,"Cooling_Medium_Air":0,"Cooling_Medium_Furnace":0,"Cooling_Medium_Salt Bath":1},
}


def build_feature_vector(process_key, C, Si, Mn, P, S, Ni, Cr, Cu, Mo,
                          ht_temp, soak_time, cool_medium,
                          t_temp=0.0, t_time=0.0):
    """Assemble feature dict matching the training schema."""
    CE   = calc_carbon_equiv(C, Mn, Si, Ni, Cr, Mo, Cu)
    A3   = calc_a3(C, Mn, Si, Ni, Cr, Mo)
    HJ   = calc_hollomon_jaffe(t_temp, t_time)
    feat = {
        "C": C, "Si": Si, "Mn": Mn, "P": P, "S": S,
        "Ni": Ni, "Cr": Cr, "Cu": Cu, "Mo": Mo,
        "Carbon_Equiv":   CE,
        "A3_Temp_C":      A3,
        "Delta_HT_A3":    ht_temp - A3,
        "Hollomon_Jaffe": HJ,
        "C_x_Cr":         C * Cr,
        "HT_Temp_C":      ht_temp,
        "Soaking_Time_min": soak_time,
        "Tempering_Temp_C":   t_temp,
        "Tempering_Time_min": t_time,
        **PROCESS_DUMMIES.get(process_key, PROCESS_DUMMIES["Quench_Temper"]),
        **MEDIUM_DUMMIES.get(cool_medium, MEDIUM_DUMMIES["Air"]),
    }
    return feat


def predict(feat_dict):
    if FEATURE_COLS:
        cols = FEATURE_COLS
    else:
        cols = list(feat_dict.keys())
    # Fill any missing one-hot columns with 0
    for c in cols:
        if c not in feat_dict:
            feat_dict[c] = 0
    df_in = pd.DataFrame([feat_dict])[cols].astype(float)

    if MODEL_MODE == "per_target":
        return {t: max(0.0, float(model_pack["models"][t].predict(df_in)[0]))
                for t in TARGETS}
    else:
        raw = model_pack["models"].predict(df_in)[0]
        return {t: max(0.0, float(v)) for t, v in zip(TARGETS[:3], raw)}


def check_ood(feat_dict):
    return [f"**{k}** = {feat_dict[k]:.3f}  (training: [{lo:.2f}, {hi:.2f}])"
            for k, (lo, hi) in TRAINING_RANGES.items()
            if k in feat_dict and
            (feat_dict[k] < lo or feat_dict[k] > hi)]


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTLY CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════════
_BASE = dict(
    plot_bgcolor ="rgba(8,13,30,1)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c8d8ff", family="Inter, sans-serif", size=11),
    margin=dict(l=55, r=20, t=50, b=50),
)


def build_gauges(preds):
    """5 gauge indicators side by side."""
    specs = [[{"type":"indicator"}]*5]
    fig   = make_subplots(rows=1, cols=5, specs=specs)
    for i, tgt in enumerate(TARGETS, 1):
        label, unit, icon, col = TARGET_LABELS[tgt]
        lo, hi = PROP_RANGES[tgt]
        val    = preds[tgt]
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=val,
            number=dict(font=dict(size=24, color=col, family="Rajdhani, sans-serif"),
                        suffix=f" {unit}"),
            title=dict(text=f"<b>{label}</b>", font=dict(size=10, color="#c8d8ff")),
            gauge=dict(
                axis=dict(range=[lo, hi], tickwidth=1,
                          tickcolor="rgba(200,216,255,0.3)",
                          tickfont=dict(size=8, color="rgba(200,216,255,0.4)")),
                bar=dict(color=col, thickness=0.25),
                bgcolor="rgba(8,13,30,0)", borderwidth=0,
                steps=[
                    dict(range=[lo, lo+(hi-lo)*0.33], color="rgba(255,255,255,0.03)"),
                    dict(range=[lo+(hi-lo)*0.33, lo+(hi-lo)*0.66], color="rgba(255,255,255,0.05)"),
                    dict(range=[lo+(hi-lo)*0.66, hi], color="rgba(255,255,255,0.08)"),
                ],
                threshold=dict(line=dict(color="white", width=2),
                               thickness=0.75, value=val),
            ),
        ), row=1, col=i)
    fig.update_layout(**_BASE, height=250,
                      margin=dict(l=20, r=20, t=40, b=10))
    return fig


def build_radar(preds):
    """Normalised 5-axis spider chart."""
    cats, vals = [], []
    for tgt in TARGETS:
        lo, hi = PROP_RANGES[tgt]
        cats.append(TARGET_LABELS[tgt][0])
        vals.append(float(np.clip((preds[tgt] - lo) / (hi - lo) * 100, 0, 100)))
    cats_closed = cats + [cats[0]]
    vals_closed  = vals + [vals[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_closed, theta=cats_closed, fill="toself",
        fillcolor="rgba(76,114,176,0.22)",
        line=dict(color="#4c72b0", width=2.5),
        marker=dict(size=7, color="#64b5f6"),
        name="Predicted",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[50]*6, theta=cats_closed, fill="toself",
        fillcolor="rgba(255,255,255,0.03)",
        line=dict(color="rgba(255,255,255,0.18)", width=1, dash="dot"),
        name="Dataset avg.",
    ))
    fig.update_layout(
        **_BASE,
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
        height=320, margin=dict(l=40,r=40,t=30,b=20),
    )
    return fig


def build_phase_diagram(C_val, ht_temp, t_temp, process_key, cool_medium):
    """Interactive Fe-C phase diagram with current operating point."""
    A1=723; A3_FE=912; PERIT=1495; MELT=1538; EUTE=1147
    C_EUT=0.76; C_AUS=2.11; MAX_C=2.3

    def a3(c): return A3_FE - (A3_FE-A1)/C_EUT * min(c,C_EUT)
    def acm(c): return A1 + (EUTE-A1)/(C_AUS-C_EUT)*(c-C_EUT)

    cc = np.linspace(0, MAX_C, 300)

    fig = go.Figure()

    # ── Phase regions ─────────────────────────────────────────────────────
    regions = [
        # (x_coords, y_coords, fill_color, label)
        ([0, 0.022, 0.022, 0],
         [0, 0, A1, A1],
         "rgba(80,200,120,0.18)", "Ferrite (α)"),
        ([0.022, C_EUT, C_EUT, 0.022],
         [0, 0, A1, A1],
         "rgba(86,180,211,0.18)", "Ferrite+Pearlite"),
        ([C_EUT, MAX_C, MAX_C, C_EUT],
         [0, 0, A1, A1],
         "rgba(176,122,214,0.18)", "Pearlite+Fe₃C"),
        ([0, C_EUT, 0],
         [A3_FE, A1, A1],
         "rgba(255,140,48,0.18)", "γ+α"),
        ([C_EUT, MAX_C, MAX_C, C_AUS],
         [A1, A1, EUTE, EUTE],
         "rgba(255,92,106,0.18)", "γ+Fe₃C"),
        ([0, 0.17, C_AUS, C_EUT, 0],
         [MELT, PERIT, EUTE, A1, A3_FE],
         "rgba(255,215,0,0.12)", "Austenite (γ)"),
        ([0, MAX_C, MAX_C, C_AUS, 0.17, 0],
         [MELT, EUTE, 1600, 1600, PERIT, MELT],
         "rgba(240,230,140,0.12)", "Liquid"),
    ]
    for xs, ys, fill, lbl in regions:
        fig.add_trace(go.Scatter(
            x=xs, y=ys, fill="toself", fillcolor=fill,
            line=dict(width=0), mode="lines",
            hovertemplate=f"<b>{lbl}</b><extra></extra>",
            showlegend=False,
        ))

    # ── Boundary lines ────────────────────────────────────────────────────
    a3_c = np.linspace(0, C_EUT, 120)
    fig.add_trace(go.Scatter(x=a3_c, y=[a3(c) for c in a3_c],
        line=dict(color="#f0a030", width=1.8, dash="dash"),
        mode="lines", name="A₃ line", showlegend=True))

    acm_c = np.linspace(C_EUT, MAX_C, 120)
    fig.add_trace(go.Scatter(x=acm_c, y=[acm(c) for c in acm_c],
        line=dict(color="#e05050", width=1.8, dash="dash"),
        mode="lines", name="Acm line", showlegend=True))

    fig.add_trace(go.Scatter(x=[0, MAX_C], y=[A1, A1],
        line=dict(color="#a0c0ff", width=1.5, dash="dot"),
        mode="lines", name="A₁ line", showlegend=True))

    # ── Eutectoid marker ──────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[C_EUT], y=[A1],
        mode="markers+text",
        marker=dict(size=8, color="#ffd700", symbol="diamond"),
        text=["S (0.76%, 723°C)"], textposition="top right",
        textfont=dict(size=9, color="#ffd700"),
        name="Eutectoid", showlegend=True,
    ))

    # ── HT operating lines ────────────────────────────────────────────────
    line_configs = [
        (ht_temp, "#509bff", "Austenitize"),
    ]
    if t_temp > 0:
        line_configs.append((t_temp, "#ffb932", "Temper"))

    for temp, col, lbl in line_configs:
        fig.add_hline(y=temp, line=dict(color=col, width=1.5, dash="dashdot"),
                      annotation_text=f"{lbl} {temp:.0f}°C",
                      annotation_font=dict(color=col, size=10),
                      annotation_position="right")

    # ── Carbon vertical line ──────────────────────────────────────────────
    fig.add_vline(x=C_val, line=dict(color="#50e3c2", width=1.5, dash="dot"),
                  annotation_text=f"C={C_val:.2f}%",
                  annotation_font=dict(color="#50e3c2", size=10))

    # ── Operating point marker ────────────────────────────────────────────
    op_phase = get_phase(C_val, ht_temp)
    op_color = PHASE_COLORS.get(op_phase, "#ffffff")
    fig.add_trace(go.Scatter(
        x=[C_val], y=[ht_temp],
        mode="markers",
        marker=dict(size=14, color=op_color,
                    line=dict(color="white", width=2),
                    symbol="circle"),
        name=f"Operating: {op_phase}",
        hovertemplate=(f"<b>Austenitize point</b><br>"
                       f"C = {C_val:.3f} wt%<br>"
                       f"T = {ht_temp:.0f} °C<br>"
                       f"Phase: {op_phase}<extra></extra>"),
        showlegend=True,
    ))

    # ── Phase region text labels ──────────────────────────────────────────
    label_positions = [
        (0.011, 350,  "α"),
        (0.39,  350,  "α + P"),
        (1.5,   350,  "P + Fe₃C"),
        (0.25,  780,  "γ + α"),
        (1.8,   900,  "γ + Fe₃C"),
        (0.40, 1100,  "γ"),
        (1.00, 1350,  "Liquid"),
    ]
    for lx, ly, ltxt in label_positions:
        fig.add_annotation(x=lx, y=ly, text=ltxt, showarrow=False,
                           font=dict(size=10, color="rgba(220,230,255,0.55)"),
                           bgcolor="rgba(0,0,0,0)")

    fig.update_layout(
        **_BASE,
        title=dict(text="<b>Fe-C Phase Diagram</b>  — Operating Conditions",
                   font=dict(size=13), x=0.5, xanchor="center"),
        xaxis=dict(title="Carbon Content (wt%)", range=[0, MAX_C],
                   gridcolor="rgba(200,216,255,0.07)", zeroline=False,
                   tickfont=dict(size=10)),
        yaxis=dict(title="Temperature (°C)", range=[0, 1600],
                   gridcolor="rgba(200,216,255,0.07)", zeroline=False,
                   tickfont=dict(size=10)),
        legend=dict(bgcolor="rgba(8,13,30,0.7)", bordercolor="rgba(76,114,176,0.3)",
                    borderwidth=1, font=dict(size=9), x=0.01, y=0.99),
        height=560,
    )
    return fig


def build_tt_profile(process_key, ht_temp, t_temp, soak_time, t_time):
    """Temperature-time schematic for the selected process."""
    PROCESS_PROFILES = {
        "Quench_Temper": {
            "color": "#4c72b0",
            "label": "Quench & Temper",
            "stages": [
                ("Preheat",      "ramp",  0,   10,  25,    500),
                ("Heat to Aust", "ramp",  10,  30,  500,   ht_temp),
                ("Soak",         "flat",  30,  30+soak_time/60, ht_temp, ht_temp),
                ("Quench",       "ramp",  30+soak_time/60, 35+soak_time/60, ht_temp, 50),
                ("Reheat temper","ramp",  35+soak_time/60, 55+soak_time/60, 50, t_temp),
                ("Temper soak",  "flat",  55+soak_time/60, 55+soak_time/60+t_time/60, t_temp, t_temp),
                ("Air cool",     "ramp",  55+soak_time/60+t_time/60, 75+soak_time/60+t_time/60, t_temp, 25),
            ]
        },
        "Normalizing": {
            "color": "#55a868",
            "label": "Normalizing",
            "stages": [
                ("Heat",      "ramp", 0,  25,  25,     ht_temp),
                ("Soak",      "flat", 25, 25+soak_time/60, ht_temp, ht_temp),
                ("Air cool",  "ramp", 25+soak_time/60, 75+soak_time/60, ht_temp, 25),
            ]
        },
        "Annealing": {
            "color": "#dd8452",
            "label": "Full Annealing",
            "stages": [
                ("Heat",         "ramp", 0,  25,  25,      ht_temp),
                ("Soak",         "flat", 25, 25+soak_time/60, ht_temp, ht_temp),
                ("Furnace cool", "ramp", 25+soak_time/60, 125+soak_time/60, ht_temp, 25),
            ]
        },
        "Stress_Relief": {
            "color": "#c44e52",
            "label": "Stress Relief",
            "stages": [
                ("Heat",     "ramp", 0,  15,  25,     ht_temp),
                ("Soak",     "flat", 15, 15+soak_time/60, ht_temp, ht_temp),
                ("Air cool", "ramp", 15+soak_time/60, 45+soak_time/60, ht_temp, 25),
            ]
        },
    }
    prof  = PROCESS_PROFILES.get(process_key, PROCESS_PROFILES["Quench_Temper"])
    color = prof["color"]
    times, temps = [0], [25]
    for _, _type, t1, t2, T1, T2 in prof["stages"]:
        if _type == "ramp":
            times += [t2]; temps += [T2]
        else:
            times += [t2]; temps += [T2]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=temps, mode="lines",
        line=dict(color=color, width=3),
        fill="tozeroy", fillcolor=f"{color}20",
        hovertemplate="t = %{x:.1f} min<br>T = %{y:.0f} °C<extra></extra>",
        name=prof["label"],
    ))
    # A1 and A3 reference lines
    A1_val = 723
    A3_val = calc_a3(0.35)  # approximate for mid-carbon
    fig.add_hline(y=A1_val, line=dict(color="#a0c0ff", width=1, dash="dot"),
                  annotation_text=f"A₁ = {A1_val}°C",
                  annotation_font=dict(color="#a0c0ff", size=9))
    if ht_temp > A3_val:
        fig.add_hline(y=A3_val, line=dict(color="#f0a030", width=1, dash="dot"),
                      annotation_text=f"A₃ ≈ {A3_val:.0f}°C",
                      annotation_font=dict(color="#f0a030", size=9))

    fig.update_layout(
        **_BASE,
        title=dict(text=f"<b>Temperature-Time Profile</b>  — {prof['label']}",
                   font=dict(size=13), x=0.5, xanchor="center"),
        xaxis=dict(title="Time (min)", gridcolor="rgba(200,216,255,0.07)"),
        yaxis=dict(title="Temperature (°C)", range=[0, max(ht_temp*1.1, 200)],
                   gridcolor="rgba(200,216,255,0.07)"),
        height=360,
    )
    return fig


def steel_grade(tensile):
    if tensile < 600:   return "Low Strength",        "#56b4d3"
    if tensile < 900:   return "Medium Strength",     "#50c878"
    if tensile < 1200:  return "High Strength",       "#ffd700"
    if tensile < 1500:  return "Very High Strength",  "#ff8c30"
    return "Ultra-High Strength", "#ff5c6a"


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem;">
  <div style="
      font-family:'Rajdhani',sans-serif;
      font-size:3.2rem;font-weight:700;
      background:linear-gradient(135deg,#4c72b0 0%,#00b4d8 40%,#56b4d3 100%);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;
      letter-spacing:0.05em;line-height:1;
  ">⚙ STEELSIGHT</div>
  <div style="color:rgba(180,200,255,0.55);font-size:0.9rem;
              letter-spacing:0.20em;text-transform:uppercase;margin-top:6px;">
      Heat Treatment · Mechanical Property Prediction · v2
  </div>
  <div style="width:80px;height:2px;
              background:linear-gradient(90deg,transparent,#4c72b0,transparent);
              margin:12px auto 0;"></div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — INPUTS
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:10px 0 14px;">
      <span style="font-family:'Rajdhani',sans-serif;font-size:1.3rem;
                   font-weight:700;color:#4c72b0;letter-spacing:0.06em;">
        INPUT PARAMETERS
      </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Process type ───────────────────────────────────────────────────────
    st.markdown("**PROCESS TYPE**")
    PROCESS_MAP = {
        "Quench & Temper":  "Quench_Temper",
        "Normalizing":      "Normalizing",
        "Full Annealing":   "Annealing",
        "Stress Relief":    "Stress_Relief",
    }
    process_label = st.selectbox(
        "Heat Treatment Process", list(PROCESS_MAP.keys()),
        label_visibility="collapsed"
    )
    process_key = PROCESS_MAP[process_label]

    st.markdown("---")

    # ── Chemical composition ───────────────────────────────────────────────
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
    st.markdown(f"""
    <div style="background:rgba(76,114,176,0.10);border:1px solid rgba(76,114,176,0.25);
                border-radius:8px;padding:8px 12px;margin:6px 0;">
      <span style="font-size:0.78rem;color:rgba(180,200,255,0.6);">Carbon Equiv (IIW): </span>
      <span style="font-family:'JetBrains Mono',monospace;color:#4c72b0;font-weight:600;">{CE:.4f}</span><br>
      <span style="font-size:0.78rem;color:rgba(180,200,255,0.6);">A₃ Temperature: </span>
      <span style="font-family:'JetBrains Mono',monospace;color:#f0a030;font-weight:600;">{A3v:.0f} °C</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Process-specific parameters (conditional) ──────────────────────────
    st.markdown(f"**{process_label.upper()} PARAMETERS**")

    if process_key == "Quench_Temper":
        ht_temp = st.slider("Austenitize Temp (°C)", 780, 980, int(max(A3v+40, 830)), 5)
        soak_time = st.slider("Soaking Time (min)", 15, 120, 45, 5)
        cool_medium = st.selectbox("Quench Medium", ["Oil", "Water", "Polymer", "Salt Bath"])
        t_temp = st.slider("Tempering Temp (°C)", 150, 700, 550, 10)
        t_time = st.slider("Tempering Time (min)", 30, 240, 120, 10)

        # A3 feedback
        if ht_temp >= A3v:
            st.success(f"✓ {ht_temp}°C > A₃ ({A3v:.0f}°C) — fully austenitized")
        else:
            st.warning(f"⚠ {ht_temp}°C < A₃ ({A3v:.0f}°C) — incomplete austenitizing")
        if t_temp >= 723:
            st.error("✗ Tempering temp ≥ A₁ — this will re-austenitize the steel!")

        HJ = calc_hollomon_jaffe(t_temp, t_time)
        st.markdown(f"""
        <div style="background:rgba(140,90,200,0.10);border:1px solid rgba(140,90,200,0.25);
                    border-radius:8px;padding:7px 12px;margin-top:6px;">
          <span style="font-size:0.76rem;color:rgba(180,200,255,0.6);">Hollomon-Jaffe H: </span>
          <span style="font-family:'JetBrains Mono',monospace;color:#8172b3;font-weight:600;">{HJ:,.0f}</span>
        </div>
        """, unsafe_allow_html=True)

    elif process_key == "Normalizing":
        ht_temp = st.slider("Normalizing Temp (°C)", 800, 980, int(max(A3v+30, 820)), 5)
        soak_time = st.slider("Soaking Time (min)", 10, 90, 30, 5)
        cool_medium = "Air"
        t_temp, t_time = 0.0, 0.0
        st.info("Cooling medium: Air  (standard normalizing)")
        if ht_temp >= A3v:
            st.success(f"✓ {ht_temp}°C > A₃ ({A3v:.0f}°C) — fully austenitized")
        else:
            st.warning(f"⚠ Below A₃ — incomplete normalizing")

    elif process_key == "Annealing":
        ht_temp = st.slider("Annealing Temp (°C)", 800, 980, int(max(A3v+30, 820)), 5)
        soak_time = st.slider("Soaking Time (min)", 30, 150, 60, 10)
        cool_medium = "Furnace"
        t_temp, t_time = 0.0, 0.0
        st.info("Cooling medium: Furnace  (controlled slow cool)")
        if ht_temp >= A3v:
            st.success(f"✓ {ht_temp}°C > A₃ ({A3v:.0f}°C) — fully austenitized")
        else:
            st.warning(f"⚠ Below A₃ — incomplete annealing")

    else:  # Stress Relief
        ht_temp = st.slider("SR Temp (°C)", 150, 700, 450, 10)
        soak_time = st.slider("Soaking Time (min)", 60, 300, 120, 15)
        cool_medium = "Air"
        t_temp, t_time = 0.0, 0.0
        st.info("Cooling medium: Air")
        if ht_temp >= 723:
            st.error("⚠ SR temp ≥ A₁ = 723°C — this is no longer stress relief!")
        elif ht_temp < 150:
            st.warning("Temperature too low for effective stress relief.")
        else:
            st.success(f"✓ {ht_temp}°C is below A₁ (723°C) — no phase change")

    st.markdown("---")

    predict_btn = st.button("⚡  PREDICT PROPERTIES", type="primary", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
if not predict_btn:
    # ── Welcome screen ───────────────────────────────────────────────────
    col_a, col_b, col_c, col_d = st.columns(4)
    for col, proc, key, color, icon, desc in [
        (col_a, "Quench & Temper",  "Quench_Temper", "#4c72b0", "🔥",
         "Austenitize → rapid quench → temper. Highest strength."),
        (col_b, "Normalizing",      "Normalizing",   "#55a868", "🌬",
         "Austenitize → air cool. Refines grain, uniform properties."),
        (col_c, "Full Annealing",   "Annealing",     "#dd8452", "🔆",
         "Austenitize → furnace cool. Maximum softness, high ductility."),
        (col_d, "Stress Relief",    "Stress_Relief", "#c44e52", "🛡",
         "Sub-A₁ heat → air cool. Relieves residual stress, no phase change."),
    ]:
        with col:
            st.markdown(html_process_card(proc, color, icon, desc),
                        unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(html_section_header(
        "How to use SteelSight",
        "Select a heat treatment process, enter alloy composition, set parameters, then click PREDICT.",
        "📖"
    ), unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div style="background:rgba(13,25,50,0.7);border:1px solid rgba(76,114,176,0.2);
                    border-radius:10px;padding:16px;">
        <div style="font-family:'Rajdhani',sans-serif;font-size:1.0rem;color:#4c72b0;
                    font-weight:700;margin-bottom:8px;">01 — SELECT PROCESS</div>
        <div style="font-size:0.80rem;color:rgba(180,200,255,0.6);">
        Choose the heat treatment method from the sidebar dropdown. The input form adapts automatically to show relevant parameters for that process.
        </div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style="background:rgba(13,25,50,0.7);border:1px solid rgba(76,114,176,0.2);
                    border-radius:10px;padding:16px;">
        <div style="font-family:'Rajdhani',sans-serif;font-size:1.0rem;color:#4c72b0;
                    font-weight:700;margin-bottom:8px;">02 — ENTER COMPOSITION</div>
        <div style="font-size:0.80rem;color:rgba(180,200,255,0.6);">
        Input the alloy chemical composition in wt%. The A₃ temperature and carbon equivalent are computed in real-time as guidance.
        </div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div style="background:rgba(13,25,50,0.7);border:1px solid rgba(76,114,176,0.2);
                    border-radius:10px;padding:16px;">
        <div style="font-family:'Rajdhani',sans-serif;font-size:1.0rem;color:#4c72b0;
                    font-weight:700;margin-bottom:8px;">03 — RUN PREDICTION</div>
        <div style="font-size:0.80rem;color:rgba(180,200,255,0.6);">
        Click PREDICT PROPERTIES to get Tensile, Yield, Hardness, Elongation and Fatigue predictions with interactive charts.
        </div></div>""", unsafe_allow_html=True)

else:
    # ── Run prediction ───────────────────────────────────────────────────
    feat = build_feature_vector(
        process_key, C, Si, Mn, P, S, Ni, Cr, Cu, Mo,
        float(ht_temp), float(soak_time), cool_medium,
        float(t_temp), float(t_time)
    )
    preds = predict(feat)

    # OOD check
    ood = check_ood(feat)
    if ood:
        with st.expander("⚠ Out-of-distribution warnings", expanded=False):
            for msg in ood:
                st.warning(msg)

    # ── Process banner ───────────────────────────────────────────────────
    PROC_COLORS = {"Quench_Temper":"#4c72b0","Normalizing":"#55a868",
                   "Annealing":"#dd8452","Stress_Relief":"#c44e52"}
    pc = PROC_COLORS.get(process_key, "#4c72b0")
    grade, gcol = steel_grade(preds["Tensile_MPa"])
    phase_at_ht = get_phase(C, ht_temp)
    phase_color = PHASE_COLORS.get(phase_at_ht, "#aaa")

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{pc}18,rgba(8,13,30,0.7));
                border:1px solid {pc}44;border-left:4px solid {pc};
                border-radius:12px;padding:14px 20px;margin-bottom:18px;
                display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
      <div>
        <span style="font-family:'Rajdhani',sans-serif;font-size:1.3rem;
                     font-weight:700;color:{pc};">{process_label}</span>
        <span style="font-size:0.78rem;color:rgba(180,200,255,0.5);margin-left:12px;">
          {ht_temp:.0f}°C · {soak_time:.0f} min · {cool_medium}
          {"  →  " + str(int(t_temp)) + "°C temper" if t_temp > 0 else ""}
        </span>
      </div>
      <div style="display:flex;gap:10px;flex-wrap:wrap;">
        <span style="background:{gcol}22;border:1px solid {gcol}55;
                     color:{gcol};border-radius:20px;padding:3px 12px;
                     font-size:0.75rem;font-weight:600;">{grade}</span>
        <span style="background:{phase_color}22;border:1px solid {phase_color}55;
                     color:{phase_color};border-radius:20px;padding:3px 12px;
                     font-size:0.75rem;font-weight:600;">{phase_at_ht}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 5 Metric cards ───────────────────────────────────────────────────
    cols = st.columns(5)
    for col, tgt in zip(cols, TARGETS):
        label, unit, icon, color = TARGET_LABELS[tgt]
        val = preds[tgt]
        if tgt == "Hardness_HB":
            disp = f"{val:.0f}"
        elif tgt == "Elongation_pct":
            disp = f"{val:.1f}"
        else:
            disp = f"{val:.0f}"
        with col:
            st.markdown(html_metric_card(label, disp, unit, color, icon),
                        unsafe_allow_html=True)

    st.markdown("<div style='margin-top:18px;'></div>", unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊  Gauges & Radar",
        "🗺  Phase Diagram",
        "📈  T-t Profile",
        "🔬  Input Summary",
    ])

    with tab1:
        st.markdown(html_section_header("Property Gauges", "", "📊"),
                    unsafe_allow_html=True)
        st.plotly_chart(build_gauges(preds), use_container_width=True)

        st.markdown("---")
        c_r, c_s = st.columns([1, 1])
        with c_r:
            st.markdown(html_section_header("Performance Radar", "Normalised 0–100%", "🕸"),
                        unsafe_allow_html=True)
            st.plotly_chart(build_radar(preds), use_container_width=True)
        with c_s:
            st.markdown(html_section_header("Numerical Results", "", "📋"),
                        unsafe_allow_html=True)
            rows = []
            for tgt in TARGETS:
                label, unit, icon, _ = TARGET_LABELS[tgt]
                rows.append({"Property": f"{icon}  {label}", "Value": f"{preds[tgt]:.1f}",
                             "Unit": unit})
            st.dataframe(pd.DataFrame(rows).set_index("Property"),
                         use_container_width=True)

            # Computed metallurgical quantities
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(html_section_header("Metallurgical Context", "", "⚗️"),
                        unsafe_allow_html=True)
            meta_rows = [
                ("Carbon Equiv (IIW)",  f"{CE:.4f}",         "wt%"),
                ("A₃ Temperature",      f"{A3v:.0f}",         "°C"),
                ("ΔT above A₃",         f"{ht_temp-A3v:+.0f}","°C"),
                ("Phase at HT temp",    phase_at_ht,          ""),
            ]
            if t_temp > 0:
                HJ = calc_hollomon_jaffe(t_temp, t_time)
                meta_rows.append(("Hollomon-Jaffe H", f"{HJ:,.0f}", ""))
            st.dataframe(
                pd.DataFrame(meta_rows, columns=["Quantity","Value","Unit"]).set_index("Quantity"),
                use_container_width=True
            )

    with tab2:
        st.plotly_chart(
            build_phase_diagram(C, float(ht_temp), float(t_temp), process_key, cool_medium),
            use_container_width=True
        )
        st.caption("The operating point (circle) marks the austenitizing temperature for your composition. "
                   "The diagram shows equilibrium phases; actual microstructure depends on cooling rate.")

    with tab3:
        st.plotly_chart(
            build_tt_profile(process_key, float(ht_temp), float(t_temp),
                             float(soak_time), float(t_time)),
            use_container_width=True
        )
        st.caption("Schematic temperature-time profile (times are approximate for illustration).")

    with tab4:
        st.markdown(html_section_header("Full Input Vector Sent to Model", "", "🔬"),
                    unsafe_allow_html=True)
        inp_df = pd.DataFrame([feat]).T.rename(columns={0: "Value"})
        inp_df["Value"] = inp_df["Value"].apply(lambda v: f"{float(v):.4f}")
        st.dataframe(inp_df, use_container_width=True)
