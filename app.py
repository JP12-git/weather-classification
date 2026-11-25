"""
Weather Intelligence | Streamlit dashboard for the ANN weather classifier.

Model artifacts (unchanged, not retrained):
    ann_model.keras, Preprocessing.pkl, LabelEncoding.pkl, Feature.pkl
Dataset:
    weather_classification_data.csv

Run:  streamlit run jp.py
"""
import html
import inspect
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Weather Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CONSTANTS: FILES, FEATURES, COLOURS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "weather_classification_data.csv"
MODEL_PATH = BASE_DIR / "ann_model.keras"
PREPROCESS_PATH = BASE_DIR / "Preprocessing.pkl"
LABEL_PATH = BASE_DIR / "LabelEncoding.pkl"
FEATURE_PATH = BASE_DIR / "Feature.pkl"

REQUIRED_FILES = [
    (DATA_PATH, "Dataset (CSV)"),
    (MODEL_PATH, "Trained ANN model"),
    (PREPROCESS_PATH, "Preprocessing pipeline"),
    (LABEL_PATH, "Label encoder"),
    (FEATURE_PATH, "Feature column names"),
]

# Column names exactly as renamed in the notebook (what Preprocessing.pkl was fitted on)
RENAME_MAP = {
    "Temperature": "Temperature (°C)",
    "Humidity": "Humidity (%)",
    "Wind Speed": "Wind Speed (km/h)",
    "Atmospheric Pressure": "Atmospheric Pressure (hpa)",
}
NUMERIC_FEATURES = [
    "Temperature (°C)", "Humidity (%)", "Wind Speed (km/h)", "Precipitation (%)",
    "Atmospheric Pressure (hpa)", "UV Index", "Visibility (km)",
]
# Default order = notebook's x.columns. Overridden by Feature.pkl when it loads.
FEATURES = [
    "Temperature (°C)", "Humidity (%)", "Wind Speed (km/h)", "Precipitation (%)",
    "Cloud Cover", "Atmospheric Pressure (hpa)", "UV Index", "Visibility (km)",
]
TARGET = "Weather Type"
CLOUD_COL = "Cloud Cover"

# Evaluation split used in the notebook: stratified 70 / 15 / 15, random_state=50
SPLIT_SEED = 50

# Palette
BG, PANEL = "#080D1B", "#0F1D38"
CYAN, VIOLET, GOLD = "#4DE3FF", "#A78BFA", "#FBBF24"
TEXT, MUTED = "#F2F6FF", "#9EAFCA"
GRID = "rgba(142,176,220,.13)"
FONT_BODY = FONT_HEAD = "Plus Jakarta Sans, Segoe UI, sans-serif"

WEATHER_COLORS = {"Sunny": "#FBBF24", "Cloudy": "#94A3B8", "Rainy": "#3B82F6", "Snowy": "#BAE6FD"}
# Deeper, high-contrast shades for dense analytics plots on the dark dashboard.
ANALYTICS_WEATHER_COLORS = {
    "Sunny": "#B88912", "Cloudy": "#64748B",
    "Rainy": "#087EAA", "Snowy": "#A9C4D2",
}
WEATHER_ICONS = {"Sunny": "☀️", "Cloudy": "☁️", "Rainy": "🌧️", "Snowy": "❄️"}
CLOUD_COLORS = {"Clear": "#4DE3FF", "Partly Cloudy": "#A78BFA", "Cloudy": "#6366F1", "Overcast": "#64748B"}

# Slider / input metadata (ranges are derived from the training data at runtime)
NUMERIC_META = [
    dict(col="Temperature (°C)", label="Temperature", unit="°C", kind="float", step=0.5, fmt="%.1f °C",
         help="Air temperature in degrees Celsius."),
    dict(col="Humidity (%)", label="Humidity", unit="%", kind="int", step=1, fmt="%d%%",
         help="Relative humidity: the percentage of moisture in the air."),
    dict(col="Wind Speed (km/h)", label="Wind Speed", unit="km/h", kind="float", step=0.5, fmt="%.1f km/h",
         help="Wind speed in kilometres per hour."),
    dict(col="Precipitation (%)", label="Precipitation", unit="%", kind="float", step=0.5, fmt="%.1f%%",
         help="Precipitation percentage as recorded in the dataset."),
    dict(col="Atmospheric Pressure (hpa)", label="Atmospheric Pressure", unit="hPa", kind="float", step=0.1,
         fmt="%.1f hPa", help="Atmospheric pressure in hectopascals. Standard sea-level pressure is about 1013 hPa."),
    dict(col="UV Index", label="UV Index", unit="", kind="int", step=1, fmt="%d",
         help="Ultraviolet index: the strength of UV radiation (0 = minimal)."),
    dict(col="Visibility (km)", label="Visibility", unit="km", kind="float", step=0.5, fmt="%.1f km",
         help="Horizontal visibility distance in kilometres."),
]
META_BY_COL = {m["col"]: m for m in NUMERIC_META}
DISPLAY_LABELS = {m["col"]: m["label"] for m in NUMERIC_META} | {CLOUD_COL: "Cloud Cover"}

# ------------------------------------------------------------
# RESULTS TAKEN FROM Weather_Prediction.ipynb  (source of truth)
# Not computed or estimated here: copied from the notebook's saved outputs.
#   * model comparison  -> cell 13 output (accuracy_score on the shared test split)
#   * ANN scores        -> cell 9 output (model.evaluate on train / val / test)
#   * training history  -> cell 7 output (Keras per-epoch log, epochs 1-79)
# ------------------------------------------------------------
NOTEBOOK_COMPARISON = {
    "Random Forest": 0.9721432632177374,
    "Support Vector Machine": 0.969300739056282,
    "Artificial Neural Network": 0.9641841955656622,
    "K-Nearest Neighbors": 0.9584991472427515,
    "Decision Tree": 0.9567936327458784,
    "Logistic Regression": 0.9516770892552586,
}
DEEP_LEARNING_MODELS = {"Artificial Neural Network"}
NOTEBOOK_ANN = {
    "train_accuracy": 0.9783059358596802, "val_accuracy": 0.9709897637367249,
    "test_accuracy": 0.9641841955656622, "test_samples": 1759,
    "epochs_run": 79, "best_epoch": 69,
}
TRAINING_HISTORY = {
    "loss": [
        0.7866, 0.4124, 0.3245, 0.2920, 0.2762, 0.2491, 0.2362, 0.2267, 0.2205, 0.2087,
        0.2021, 0.2051, 0.1926, 0.1957, 0.1895, 0.1850, 0.1702, 0.1817, 0.1767, 0.1678,
        0.1645, 0.1646, 0.1620, 0.1685, 0.1601, 0.1569, 0.1525, 0.1468, 0.1504, 0.1489,
        0.1450, 0.1438, 0.1495, 0.1362, 0.1447, 0.1340, 0.1411, 0.1440, 0.1337, 0.1381,
        0.1241, 0.1309, 0.1374, 0.1343, 0.1357, 0.1381, 0.1297, 0.1347, 0.1303, 0.1271,
        0.1247, 0.1306, 0.1238, 0.1333, 0.1230, 0.1211, 0.1253, 0.1235, 0.1221, 0.1271,
        0.1109, 0.1214, 0.1242, 0.1125, 0.1165, 0.1171, 0.1215, 0.1236, 0.1129, 0.1204,
        0.1124, 0.1192, 0.1163, 0.1210, 0.1083, 0.1117, 0.1091, 0.1152, 0.1147,
    ],
    "val_loss": [
        0.3361, 0.2150, 0.1818, 0.1676, 0.1521, 0.1434, 0.1390, 0.1348, 0.1330, 0.1279,
        0.1315, 0.1260, 0.1256, 0.1292, 0.1272, 0.1239, 0.1247, 0.1226, 0.1183, 0.1182,
        0.1182, 0.1150, 0.1152, 0.1121, 0.1159, 0.1129, 0.1119, 0.1113, 0.1085, 0.1082,
        0.1113, 0.1119, 0.1085, 0.1095, 0.1082, 0.1067, 0.1105, 0.1046, 0.1051, 0.1041,
        0.1059, 0.1058, 0.1083, 0.1098, 0.1067, 0.1062, 0.1067, 0.1042, 0.1038, 0.1093,
        0.1074, 0.1067, 0.1029, 0.1014, 0.1057, 0.1069, 0.1038, 0.1053, 0.1015, 0.1010,
        0.1034, 0.1019, 0.1047, 0.1034, 0.0997, 0.0999, 0.1008, 0.1003, 0.0991, 0.1013,
        0.1034, 0.1028, 0.1065, 0.0996, 0.1051, 0.1045, 0.1032, 0.1082, 0.1027,
    ],
    "accuracy": [
        0.7192, 0.8808, 0.9082, 0.9192, 0.9268, 0.9297, 0.9346, 0.9387, 0.9394, 0.9430,
        0.9433, 0.9415, 0.9483, 0.9460, 0.9449, 0.9491, 0.9525, 0.9505, 0.9502, 0.9520,
        0.9530, 0.9531, 0.9564, 0.9525, 0.9541, 0.9570, 0.9572, 0.9577, 0.9582, 0.9590,
        0.9594, 0.9606, 0.9582, 0.9629, 0.9588, 0.9632, 0.9592, 0.9599, 0.9627, 0.9595,
        0.9651, 0.9640, 0.9627, 0.9615, 0.9611, 0.9597, 0.9655, 0.9597, 0.9634, 0.9653,
        0.9620, 0.9651, 0.9638, 0.9604, 0.9650, 0.9672, 0.9655, 0.9660, 0.9638, 0.9621,
        0.9678, 0.9653, 0.9653, 0.9682, 0.9664, 0.9672, 0.9650, 0.9642, 0.9671, 0.9622,
        0.9681, 0.9661, 0.9673, 0.9658, 0.9683, 0.9664, 0.9692, 0.9672, 0.9655,
    ],
    "val_accuracy": [
        0.9266, 0.9386, 0.9522, 0.9539, 0.9585, 0.9602, 0.9596, 0.9590, 0.9619, 0.9659,
        0.9625, 0.9642, 0.9653, 0.9613, 0.9636, 0.9630, 0.9642, 0.9659, 0.9636, 0.9659,
        0.9653, 0.9653, 0.9647, 0.9670, 0.9636, 0.9647, 0.9647, 0.9659, 0.9681, 0.9687,
        0.9681, 0.9664, 0.9664, 0.9664, 0.9659, 0.9647, 0.9659, 0.9653, 0.9664, 0.9664,
        0.9647, 0.9636, 0.9625, 0.9636, 0.9625, 0.9670, 0.9670, 0.9681, 0.9664, 0.9636,
        0.9687, 0.9676, 0.9642, 0.9659, 0.9642, 0.9630, 0.9647, 0.9676, 0.9670, 0.9642,
        0.9670, 0.9670, 0.9653, 0.9664, 0.9659, 0.9659, 0.9676, 0.9670, 0.9710, 0.9681,
        0.9670, 0.9676, 0.9664, 0.9647, 0.9670, 0.9647, 0.9670, 0.9664, 0.9681,
    ],
}

# ============================================================
# GLOBAL STYLING (single stylesheet)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
:root{--bg:#080D1B;--bg2:#0B1428;--panel:rgba(18,30,56,.72);--solid:#0F1D38;--head:#16274A;
--border:rgba(132,174,231,.16);--cyan:#4DE3FF;--violet:#A78BFA;--gold:#FBBF24;--text:#F2F6FF;--muted:#9EAFCA;}
.stApp{font-family:'Plus Jakarta Sans','Segoe UI',system-ui,sans-serif;color:var(--text);
background:radial-gradient(ellipse at 12% 4%,rgba(55,85,180,.20),transparent 36%),
radial-gradient(ellipse at 88% 16%,rgba(125,65,190,.15),transparent 32%),
linear-gradient(135deg,#080D1B 0%,#0B1428 55%,#0E1228 100%);}
.block-container{max-width:1380px;padding-top:1.2rem;padding-bottom:2.5rem;}
h1,h2,h3,h4{font-family:'Plus Jakarta Sans','Segoe UI',system-ui,sans-serif !important;color:var(--text) !important;font-weight:700 !important;letter-spacing:-.015em;}
h1{font-weight:800 !important;} h3{font-size:1.25rem !important;}
[data-testid="stWidgetLabel"] p,label{color:var(--text) !important;font-weight:500;}
[data-testid="stCaptionContainer"],.stCaption{color:var(--muted) !important;}
#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"]{visibility:hidden;display:none;}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0E182E,#09111F);border-right:1px solid var(--border);}
[data-testid="stSidebar"] *{color:#EAF1FF;}
hr{border-color:var(--border);}
/* navigation (st.radio styled as pills) */
div[role="radiogroup"]{gap:8px;flex-wrap:wrap;}
div[role="radiogroup"]>label{background:rgba(15,29,56,.8);border:1px solid var(--border);border-radius:999px;
padding:7px 18px;margin:0;transition:all .18s ease;cursor:pointer;}
div[role="radiogroup"]>label>div:first-child{display:none;}
div[role="radiogroup"]>label:hover{border-color:var(--cyan);}
div[role="radiogroup"]>label:has(input:checked){background:linear-gradient(100deg,rgba(31,126,169,.55),rgba(104,81,194,.55));
border-color:rgba(77,227,255,.6);}
div[role="radiogroup"]>label p{font-weight:600;font-size:.9rem;}
/* hero */
.hero{position:relative;overflow:hidden;border:1px solid rgba(112,194,255,.22);border-radius:22px;padding:28px 32px;margin:4px 0 18px;
background:linear-gradient(115deg,rgba(25,56,95,.9),rgba(50,39,100,.78) 60%,rgba(21,39,72,.9));box-shadow:0 14px 40px rgba(0,0,0,.22);}
.hero h1{font-size:clamp(1.8rem,3.6vw,2.9rem);margin:.3rem 0 .4rem;}
.hero p{color:#C4D3EA;max-width:720px;margin:0;line-height:1.55;}
.hero-orb{position:absolute;right:6%;top:22px;font-size:4.6rem;opacity:.7;}
.eyebrow,.kicker{color:#75E8FF;font-size:.72rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;}
.sec-title{font-family:'Plus Jakarta Sans';font-weight:700;font-size:1.05rem;margin:.2rem 0 .6rem;}
/* cards */
.card{background:linear-gradient(145deg,rgba(25,42,72,.85),rgba(15,26,49,.8));border:1px solid var(--border);border-radius:16px;
padding:16px 18px;box-shadow:0 8px 24px rgba(0,0,0,.16);min-height:100px;transition:border-color .2s ease,transform .2s ease;}
.card:hover{border-color:rgba(77,227,255,.4);transform:translateY(-2px);}
.m-label{color:#A9BBD6;font-size:.72rem;text-transform:uppercase;letter-spacing:.09em;font-weight:600;}
.m-value{font-family:'Plus Jakarta Sans';font-size:1.7rem;font-weight:700;margin-top:6px;}
.m-note{color:#8EA4C4;font-size:.78rem;margin-top:2px;}
.panel{background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:16px 20px;line-height:1.6;}
.panel h3{margin:0 0 .5rem;}
.muted{color:var(--muted);font-size:.88rem;}
.result{background:linear-gradient(135deg,rgba(23,55,83,.95),rgba(49,39,91,.88));border:1px solid rgba(77,227,255,.3);border-radius:20px;padding:22px;}
.badge{display:inline-flex;align-items:center;gap:6px;border-radius:999px;padding:6px 14px;font-weight:600;font-size:.88rem;margin:0 8px 8px 0;border:1px solid;}
/* current-input grid */
.in-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:6px 0 14px;}
.in-chip{background:rgba(17,30,55,.8);border:1px solid var(--border);border-radius:12px;padding:10px 14px;}
.in-chip .k{color:var(--muted);font-size:.72rem;font-weight:500;}
.in-chip .v{font-family:'Plus Jakarta Sans';font-weight:700;font-size:1.08rem;color:var(--cyan);margin-top:2px;}
/* tables */
.tbl-wrap{width:100%;overflow-x:hidden;border:1px solid var(--border);border-radius:14px;background:var(--solid);margin-bottom:10px;}
.tbl{border-collapse:collapse;width:100%;table-layout:fixed;font-size:.84rem;}
.tbl th{position:sticky;top:0;background:var(--head);color:#DDE8FF;text-align:left;font-weight:600;padding:11px 10px;white-space:normal;z-index:1;}
.tbl td{padding:11px 10px;border-top:1px solid rgba(132,174,231,.10);color:var(--text);white-space:normal;overflow-wrap:anywhere;}
.tbl td.num{color:#8FEFFF;text-align:right;font-variant-numeric:tabular-nums;}
.tbl th.r{text-align:right;}
.tbl tbody tr:hover td{background:rgba(77,227,255,.05);}
.tbl tr.avg td{background:rgba(77,227,255,.07);font-weight:600;}
.report-table th:first-child,.report-table td:first-child{width:27%;}
.report-table th:last-child,.report-table td:last-child{width:13%;text-align:right;}
.report-table td.metric{text-align:center;color:#DDEBFF;font-weight:650;font-variant-numeric:tabular-nums;}
@media(max-width:700px){.tbl{font-size:.73rem}.tbl th,.tbl td{padding:8px 5px}.report-table th:first-child,.report-table td:first-child{width:30%;}}
/* about */
.kv{display:grid;grid-template-columns:150px 1fr;gap:8px 12px;font-size:.92rem;}
.kv .k{color:var(--muted);} .kv .v{font-weight:500;}
.pipe{position:relative;display:flex;flex-direction:column;gap:12px;}
.step{position:relative;display:flex;gap:14px;align-items:flex-start;background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:13px 16px;}
.step:not(:last-child)::after{content:"";position:absolute;left:30px;bottom:-13px;width:2px;height:13px;background:rgba(77,227,255,.35);}
.step .n{flex:0 0 32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-family:'Plus Jakarta Sans';font-weight:700;
background:linear-gradient(100deg,#167DAA,#6556C9);}
.step b{font-family:'Plus Jakarta Sans';} .step span{display:block;color:var(--muted);font-size:.86rem;margin-top:2px;}
.foot{text-align:center;color:#8197B7;font-size:.82rem;line-height:1.7;padding:10px 0 4px;}
.foot b{color:#C9D6EC;font-family:'Plus Jakarta Sans';font-weight:600;}
/* widgets */
.stButton>button{background:linear-gradient(100deg,#167DAA,#6556C9);color:#fff;border:1px solid rgba(130,224,255,.25);
border-radius:12px;padding:.7rem 1.2rem;font-weight:700;transition:all .18s ease;}
.stButton>button:hover{transform:translateY(-1px);border-color:#69E7FF;color:#fff;box-shadow:0 6px 20px rgba(55,190,240,.2);}
div[data-baseweb="select"]>div{background:rgba(15,28,52,.9);border-color:rgba(125,170,230,.25);border-radius:10px;}
div[data-baseweb="select"] *{color:var(--text);}
[data-baseweb="popover"] [role="listbox"]{background:#0F1D38;}
[data-baseweb="popover"] li{color:#F2F6FF;}
span[data-baseweb="tag"]{background:rgba(77,227,255,.18);}
div[data-baseweb="slider"] div[style*="linear-gradient"]{background:linear-gradient(90deg,#4DE3FF,#A78BFA) !important;}
div[data-baseweb="slider"] [role="slider"]{background:#4DE3FF !important;border:2px solid #080D1B !important;box-shadow:0 0 0 3px rgba(77,227,255,.25) !important;}
[data-testid="stSliderThumbValue"]{color:#4DE3FF !important;font-family:'Plus Jakarta Sans';font-weight:700;}
[data-testid="stSliderTickBarMin"],[data-testid="stSliderTickBarMax"]{color:var(--muted) !important;}
div[data-testid="stPlotlyChart"]{background:rgba(17,29,53,.6);border:1px solid rgba(123,166,222,.13);border-radius:16px;padding:6px;}
.stTabs [data-baseweb="tab-list"]{gap:8px;}
@media (max-width:768px){.hero{padding:20px;} .hero-orb{display:none;} .kv{grid-template-columns:1fr;} .m-value{font-size:1.4rem;}}

/* ===== Typography: Plus Jakarta Sans everywhere (icon fonts and code are excluded) ===== */
html{font-size:16px;}
.stApp,.stApp *:not([data-testid="stIconMaterial"]):not([class*="material" i]):not(code):not(pre):not(kbd){font-family:'Plus Jakarta Sans','Segoe UI',system-ui,sans-serif !important;}
/* ===== Motion + live shadows ===== */
@keyframes fadeUp{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
@keyframes aurora{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
@keyframes sheen{from{transform:translateX(-130%)}to{transform:translateX(230%)}}
@keyframes breathe{0%,100%{box-shadow:0 0 0 1px rgba(77,227,255,.20),0 12px 30px rgba(0,0,0,.28),0 0 22px rgba(77,227,255,.14)}
50%{box-shadow:0 0 0 1px rgba(167,139,250,.34),0 12px 30px rgba(0,0,0,.28),0 0 42px rgba(167,139,250,.30)}}
@keyframes btnPulse{0%,100%{box-shadow:0 0 0 0 rgba(77,227,255,.38)}50%{box-shadow:0 0 0 9px rgba(77,227,255,0)}}
.hero{background-size:220% 220% !important;animation:aurora 18s ease-in-out infinite,fadeUp .6s ease backwards;
box-shadow:0 14px 44px rgba(0,0,0,.28),0 0 30px rgba(77,227,255,.10);}
.hero::after{content:"";position:absolute;inset:0;pointer-events:none;
background:linear-gradient(110deg,transparent 35%,rgba(255,255,255,.07) 50%,transparent 65%);animation:sheen 8s ease-in-out infinite;}
.card,.panel,.step,.tbl-wrap{animation:fadeUp .55s ease backwards;transition:box-shadow .3s ease,border-color .3s ease,transform .3s ease;}
[data-testid="stColumn"]:nth-child(2) .card,[data-testid="column"]:nth-child(2) .card{animation-delay:.07s;}
[data-testid="stColumn"]:nth-child(3) .card,[data-testid="column"]:nth-child(3) .card{animation-delay:.14s;}
[data-testid="stColumn"]:nth-child(4) .card,[data-testid="column"]:nth-child(4) .card{animation-delay:.21s;}
.card:hover,.panel:hover,.step:hover{border-color:rgba(77,227,255,.45);box-shadow:0 14px 34px rgba(0,0,0,.32),0 0 26px rgba(77,227,255,.16);}
.in-chip{transition:box-shadow .25s ease,border-color .25s ease,transform .25s ease;}
.in-chip:hover{border-color:rgba(77,227,255,.5);box-shadow:0 0 18px rgba(77,227,255,.20);transform:translateY(-2px);}
.result{animation:fadeUp .55s ease backwards,breathe 5s ease-in-out .6s infinite;}
div[data-testid="stPlotlyChart"]{animation:fadeUp .6s ease backwards;transition:box-shadow .3s ease,border-color .3s ease;}
div[data-testid="stPlotlyChart"]:hover{border-color:rgba(77,227,255,.38);box-shadow:0 10px 30px rgba(0,0,0,.30),0 0 24px rgba(77,227,255,.12);}
.stButton>button{position:relative;overflow:hidden;animation:btnPulse 3.2s ease-in-out infinite;}
.stButton>button::after{content:"";position:absolute;inset:0;pointer-events:none;
background:linear-gradient(110deg,transparent 35%,rgba(255,255,255,.22) 50%,transparent 65%);animation:sheen 4.8s ease-in-out infinite;}
div[role="radiogroup"]>label:has(input:checked){box-shadow:0 0 18px rgba(77,227,255,.30);}
.reason-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:12px 0;}
.reason-grid .card{min-height:0;}
.reason-grid .card h4{margin:.35rem 0 .4rem;font-size:1.08rem;color:var(--cyan) !important;}
.reason-grid .card p{margin:0;color:var(--muted);font-size:.95rem;line-height:1.55;}
@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation:none !important;transition:none !important;}}
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA + MODEL HELPERS
# ============================================================
@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        return None
    # Same renaming as the notebook so the saved preprocessor sees the column names it was fitted on
    return pd.read_csv(DATA_PATH).rename(columns=RENAME_MAP)


def clean_training_data(data):
    """Reproduce the notebook's outlier filtering, column dropping and feature/target split."""
    work = data.copy()
    # (column, lower IQR factor, upper IQR factor), applied sequentially exactly as in the notebook
    for col, lower_factor, upper_factor in [
        ("Temperature (°C)", 1.5, 1.4),
        ("Wind Speed (km/h)", 1.5, 1.4),
        ("Atmospheric Pressure (hpa)", 1.35, 1.35),
    ]:
        q1, q3 = work[col].quantile(.25), work[col].quantile(.75)
        iqr = q3 - q1
        work = work[(work[col] > q1 - lower_factor * iqr) & (work[col] < q3 + upper_factor * iqr)]
    work = work.drop(columns=["Location", "Season"], errors="ignore")
    return work.drop(columns=[TARGET]), work[TARGET].copy(), work


def to_dense(array):
    """ColumnTransformer may return a sparse matrix; Keras needs a dense array."""
    return array.toarray() if hasattr(array, "toarray") else np.asarray(array)


@st.cache_data
def get_input_domain(data):
    """Slider ranges = min/max of the outlier-filtered training data (the model's supported domain).
    Bounds are rounded *inward* to the slider step so no value outside the training range is offered."""
    X, _, _ = clean_training_data(data)
    domain = {}
    for meta in NUMERIC_META:
        col, step = meta["col"], meta["step"]
        lo, hi, med = X[col].min(), X[col].max(), X[col].median()
        if meta["kind"] == "int":
            lo, hi, med = int(math.ceil(lo)), int(math.floor(hi)), int(round(med))
        else:
            lo = round(math.ceil(lo / step - 1e-9) * step, 4)
            hi = round(math.floor(hi / step + 1e-9) * step, 4)
            med = round(round(med / step) * step, 4)
            lo, hi, med = float(lo), float(hi), float(med)
        domain[col] = (lo, hi, min(max(med, lo), hi))
    return domain


@st.cache_resource
def load_model_artifacts():
    """Returns (model, preprocessor, classes, feature_cols, problems).
    `problems` holds human-readable messages (missing files / load errors)."""
    problems = []
    missing = [p.name for p, _ in REQUIRED_FILES[1:] if not p.exists()]
    if missing:
        return None, None, None, list(FEATURES), ["Missing file(s): " + ", ".join(missing)]

    model = preprocessor = classes = None
    feature_cols = list(FEATURES)
    import joblib

    try:
        preprocessor = joblib.load(PREPROCESS_PATH)
    except ModuleNotFoundError as exc:
        problems.append(f"Preprocessing.pkl needs a missing Python package: {exc.name}. Run: pip install {exc.name}")
    except Exception as exc:
        problems.append(f"Could not load Preprocessing.pkl: {exc}")

    try:
        classes = [str(c) for c in joblib.load(LABEL_PATH).classes_]
    except ModuleNotFoundError as exc:
        problems.append(f"LabelEncoding.pkl needs a missing Python package: {exc.name}.")
    except Exception as exc:
        problems.append(f"Could not load LabelEncoding.pkl: {exc}")

    # Feature.pkl: fall back to the preprocessor's own column list if it cannot be read
    try:
        feature_cols = [str(c) for c in list(joblib.load(FEATURE_PATH))]
    except Exception:
        if preprocessor is not None and hasattr(preprocessor, "feature_names_in_"):
            feature_cols = [str(c) for c in preprocessor.feature_names_in_]

    try:
        from tensorflow.keras.models import load_model
        model = load_model(MODEL_PATH)
    except ModuleNotFoundError as exc:
        problems.append(f"TensorFlow is not installed ({exc.name}). Run: pip install tensorflow")
    except Exception as exc:
        problems.append(f"Could not load ann_model.keras: {exc}")

    if problems:
        return None, None, None, feature_cols, problems
    return model, preprocessor, classes, feature_cols, []


@st.cache_data(show_spinner="Evaluating the ANN on the train / validation / test splits…")
def compute_evaluation(_model, _preprocessor, data, classes, feature_cols, model_stamp):
    """Recreate the notebook's stratified 70/15/15 split (random_state=50) and score the saved ANN.
    `model_stamp` only invalidates the cache when ann_model.keras changes."""
    X, y_text, _ = clean_training_data(data)
    y = y_text.map({name: i for i, name in enumerate(classes)})
    X = X[list(feature_cols)]
    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SPLIT_SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=SPLIT_SEED)
    out = {}
    for name, Xs, ys in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        prob = _model.predict(to_dense(_preprocessor.transform(Xs)), verbose=0)
        pred = np.argmax(prob, axis=1)
        out[name] = {"y": ys.to_numpy(), "pred": pred, "accuracy": float(accuracy_score(ys, pred)), "n": int(len(ys))}
    return out
    
def wcolor(label):
    return WEATHER_COLORS.get(label, CYAN)


# ============================================================
# UI HELPERS
# ============================================================
def _stretch(fn):
    """Streamlit renamed use_container_width -> width='stretch'; support both."""
    try:
        return {"width": "stretch"} if "width" in inspect.signature(fn).parameters else {"use_container_width": True}
    except (TypeError, ValueError):
        return {"use_container_width": True}


STRETCH_CHART = _stretch(st.plotly_chart)
STRETCH_BUTTON = _stretch(st.button)
PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


def esc(value):
    return html.escape(str(value))


def metric_card(label, value, note=""):
    st.markdown(
        f'<div class="card"><div class="m-label">{esc(label)}</div><div class="m-value">{value}</div>'
        f'<div class="m-note">{esc(note)}</div></div>', unsafe_allow_html=True)


def page_header(kicker, title, caption=""):
    st.markdown(f'<div class="kicker">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    if caption:
        st.caption(caption)


def chart_note(text):
    st.caption(text)


def fmt_cell(v):
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return "—"
    if isinstance(v, (bool, np.bool_)):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):,.2f}".rstrip("0").rstrip(".")
    return str(v)


def render_table(df, index=False, index_label="", max_height=460):
    """Dark-themed HTML table (replaces the default Streamlit dataframe look)."""
    d = df.reset_index() if index else df
    if index and index_label:
        d = d.rename(columns={d.columns[0]: index_label})
    head = "".join(f"<th>{esc(c)}</th>" for c in d.columns)
    rows = []
    for row in d.itertuples(index=False, name=None):
        cells = []
        for v in row:
            is_num = isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))
            cells.append(f'<td class="{"num" if is_num else ""}">{esc(fmt_cell(v))}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    st.markdown(
        f'<div class="tbl-wrap" style="max-height:{max_height}px"><table class="tbl"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>', unsafe_allow_html=True)


def render_report_table(report, classes):
    """Compact classification report with plain metric values (no slider/progress bars)."""
    rows = []
    entries = [(c, report[c], False) for c in classes] + [
        ("Macro Average", report["macro avg"], True), ("Weighted Average", report["weighted avg"], True)]
    for name, r, is_avg in entries:
        dot = "" if is_avg else f'<span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:{wcolor(name)};margin-right:8px"></span>'
        rows.append(
            f'<tr class="{"avg" if is_avg else ""}"><td>{dot}{esc(name)}</td>'
            f'<td class="metric">{r["precision"]:.3f}</td>'
            f'<td class="metric">{r["recall"]:.3f}</td>'
            f'<td class="metric">{r["f1-score"]:.3f}</td>'
            f'<td class="num">{int(r["support"]):,}</td></tr>')
    st.markdown(
        '<div class="tbl-wrap"><table class="tbl report-table"><thead><tr><th>Weather Type</th><th>Precision</th><th>Recall</th>'
        '<th>F1-score</th><th class="r">Support</th></tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>',
        unsafe_allow_html=True)


def input_chips_html(values):
    chips = []
    for col in FEATURES:
        if col not in values:
            continue
        v = values[col]
        if col == CLOUD_COL:
            shown = str(v).title()
        else:
            meta = META_BY_COL[col]
            num = f"{v:.1f}" if meta["kind"] == "float" else f"{int(v)}"
            shown = f"{num} {meta['unit']}".strip()
        chips.append(f'<div class="in-chip"><div class="k">{esc(DISPLAY_LABELS.get(col, col))}</div><div class="v">{esc(shown)}</div></div>')
    return '<div class="in-grid">' + "".join(chips) + "</div>"


def render_footer():
    st.markdown(
        '<hr><div class="foot"><b>Weather Classification · Deep Learning Project · Artificial Neural Network </b><br>'
        '<b>Author : Prerak Jasani</b><br>'
        '</div>', unsafe_allow_html=True)


# ============================================================
# CHART DESIGN SYSTEM (every Plotly figure goes through style_fig)
# ============================================================
def style_fig(fig, height=380, title=None, legend=True):
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_BODY, color=MUTED, size=12),
        margin=dict(l=24, r=24, t=56 if title else 24, b=24),
        title=dict(text=title or "", font=dict(family=FONT_HEAD, size=16, color=TEXT), x=0.01, xanchor="left"),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="center", x=0.5,
                    bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT, size=12), title_text=""),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=CYAN, font=dict(family=FONT_BODY, color=TEXT, size=12)),
    )
    axis = dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED),
                title_font=dict(color=TEXT, size=12))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def show_chart(fig):
    st.plotly_chart(fig, config=PLOT_CONFIG, **STRETCH_CHART)


# ============================================================
# LOAD RESOURCES
# ============================================================
data = load_dataset()
model, preprocessor, classes, feature_cols, artifact_problems = load_model_artifacts()
if classes is None:
    classes = sorted(WEATHER_COLORS)  # only used for display when artifacts are missing

# ============================================================
# SIDEBAR + HEADER + NAVIGATION
# ============================================================

st.markdown(
    '<div class="hero"><div class="eyebrow"> DEEP LEARNING • MACHINE LEARNING • ATMOSPHERIC ANALYTICS</div>'
    '<h1>Weather Intelligence</h1>'
    '<p>Explore meteorological patterns and classify atmospheric conditions using a trained Artificial Neural Network.</p>',
     unsafe_allow_html=True)

PAGES = ["Overview", "Predict Weather", "Model Intelligence", "Weather Analytics", "Dataset Explorer", "About"]
page = st.radio("Main navigation", PAGES, horizontal=True, label_visibility="collapsed")
st.markdown("---")

if data is None:
    st.error(f"Dataset file not found. Place `{DATA_PATH.name}` in the same folder as this script: {BASE_DIR}")
    render_footer()
    st.stop()


# ============================================================
# PAGE: OVERVIEW
# ============================================================
def render_overview():
    page_header("01 / PLATFORM OVERVIEW", "Atmospheric overview"
                )
    counts = data[TARGET].value_counts()
    a, b, c, d = st.columns(4)
    with a: metric_card("DATASET RECORDS", f"{len(data):,}", "Raw dataset rows")
    with b: metric_card("INPUT FEATURES", "08", "7 numeric + Cloud Cover")
    with c: metric_card("WEATHER CLASSES", f"{data[TARGET].nunique()}", "Sunny · Cloudy · Rainy · Snowy")
    with d: metric_card("MODEL TYPE", "ANN", "Dense neural network classifier")
    st.write("")
    left, right = st.columns([1.05, 1])
    with left:
        fig = px.pie(counts.rename_axis("Weather").reset_index(name="Records"), names="Weather", values="Records",
                     hole=.62, color="Weather", color_discrete_map=WEATHER_COLORS)
        fig.update_traces(textposition="outside", textinfo="label+percent", textfont=dict(color=TEXT),
                          marker=dict(line=dict(color=BG, width=2)),
                          hovertemplate="%{label}<br>%{value:,} records (%{percent})<extra></extra>")
        fig.add_annotation(text="WEATHER<br>CLASSES", x=.5, y=.5, showarrow=False, font=dict(size=14, color=TEXT, family=FONT_HEAD))
        show_chart(style_fig(fig, 380, legend=False))
    with right:
        st.markdown("### Classification distribution")
        dist = counts.reindex(classes).dropna() if classes else counts
        for label, value in dist.items():
            st.markdown(
                f'<div class="panel" style="margin-bottom:10px;padding:12px 16px"><b>{WEATHER_ICONS.get(label, "🌤️")} {esc(label)}</b>'
                f'<span style="float:right;color:{MUTED}">{int(value):,} records · {value / len(data):.1%}</span></div>',
                unsafe_allow_html=True)
    st.markdown("### Project workflow")
    steps = [("01", "Data preparation", "Outlier filtering and feature selection"),
             ("02", "Preprocessing", "StandardScaler + OneHotEncoder"),
             ("03", "Neural network", "Dense layers with ReLU and Softmax"),
             ("04", "Classification", "Four weather categories")]
    for col, (n, title, desc) in zip(st.columns(4), steps):
        with col:
            st.markdown(f'<div class="card"><div class="kicker">{n}</div><h3 style="margin:.3rem 0">{title}</h3><div class="muted">{desc}</div></div>',
                        unsafe_allow_html=True)
    if model is None:
        st.warning("Prediction and model evaluation are unavailable because some model files are missing or could not be loaded. "
                   "Open the sidebar → Required files for details.")


# ============================================================
# PAGE: PREDICT WEATHER
# ============================================================
def make_slider(meta, domain):
    lo, hi, default = domain[meta["col"]]
    return st.slider(f"{meta['label']} ({meta['unit']})" if meta["unit"] else meta["label"],
                     min_value=lo, max_value=hi, value=default, step=meta["step"],
                     format=meta["fmt"], help=meta["help"], key=f"in_{meta['col']}")


def show_prediction(pred):
    label, conf, probs = pred["label"], pred["confidence"], pred["probs"]
    color = wcolor(label)
    st.markdown("---")
    st.markdown("### Classification result")
    left, right = st.columns([.85, 1.15], gap="large")
    with left:
        st.markdown(
            f'<div class="result" style="border-color:{color}88"><div class="kicker">PREDICTED WEATHER TYPE</div>'
            f'<div style="font-size:4rem;margin:10px 0">{WEATHER_ICONS.get(label, "🌤️")}</div>'
            f'<h2 style="font-size:2rem;margin:0;color:{color} !important">{esc(label)}</h2>'
            f'<div style="font-family:Plus Jakarta Sans,Segoe UI,sans-serif;font-size:2.1rem;font-weight:700;color:{CYAN};margin-top:6px">{conf:.1%}</div>'
            f'<div class="muted">Model confidence for the predicted class</div></div>', unsafe_allow_html=True)
    with right:
        order = np.argsort(probs)  # ascending so the largest bar sits on top
        names = [classes[i] for i in order]
        top = pred["index"]
        fig = go.Figure(go.Bar(
            x=[float(probs[i]) for i in order], y=names, orientation="h",
            marker=dict(color=[wcolor(n) for n in names],
                        opacity=[1.0 if i == top else 0.45 for i in order],
                        line=dict(color=[TEXT if i == top else "rgba(0,0,0,0)" for i in order],
                                  width=[2 if i == top else 0 for i in order])),
            text=[f"{probs[i]:.1%}" for i in order], textposition="outside", cliponaxis=False,
            textfont=dict(color=TEXT), hovertemplate="%{y}: %{x:.2%}<extra></extra>"))
        fig.update_xaxes(range=[0, 1.12], tickformat=".0%", title="Model probability")
        show_chart(style_fig(fig, 300, title="Probability distribution", legend=False))
    # if conf < 0.60:
    #     st.warning("The top probability is fairly low, so these conditions sit between weather types. Treat the result with extra caution.")


def render_predict():
    page_header("02 / AI CLASSIFICATION ENGINE", "Predict atmospheric conditions",
                "Adjust the eight measurements used by the ANN, then run the analysis.")
    if model is None:
        st.error("The trained ANN artifacts are not available.")
        for path, label in REQUIRED_FILES[1:]:
            st.write(f"{'✅' if path.exists() else '❌'} `{path.name}`: {label}")
        for problem in artifact_problems:
            st.error(problem)
        st.info(f"Place all of the files above in: {BASE_DIR}")
        return

    domain = get_input_domain(data)
    st.markdown("#### Atmospheric measurements")
    values = {}
    left, right = st.columns(2, gap="large")
    with left:
        for col in ["Temperature (°C)", "Humidity (%)", "Wind Speed (km/h)", "Precipitation (%)"]:
            values[col] = make_slider(META_BY_COL[col], domain)
    with right:
        cloud_options = sorted(data[CLOUD_COL].dropna().unique().tolist())
        values[CLOUD_COL] = st.selectbox(
            "Cloud Cover", cloud_options, format_func=str.title, key="in_cloud",
            index=cloud_options.index("partly cloudy") if "partly cloudy" in cloud_options else 0,
            help="Observed cloud-cover category: clear, partly cloudy, cloudy or overcast.")
        for col in ["Atmospheric Pressure (hpa)", "UV Index", "Visibility (km)"]:
            values[col] = make_slider(META_BY_COL[col], domain)

    st.markdown("#### Current Input")
    st.markdown(input_chips_html(values), unsafe_allow_html=True)

    if st.button("✦ Analyse Weather Pattern", type="primary", key="analyse", **STRETCH_BUTTON):
        try:
            input_df = pd.DataFrame([values])[list(feature_cols)]  # same column order the preprocessor was fitted with
            probs = model.predict(to_dense(preprocessor.transform(input_df)), verbose=0)[0]
            idx = int(np.argmax(probs))
            st.session_state["prediction"] = {
                "label": classes[idx], "index": idx, "confidence": float(probs[idx]),
                "probs": probs, "inputs": dict(values)}
        except Exception as exc:
            st.session_state.pop("prediction", None)
            st.error(f"Prediction failed: {exc}")
            st.info("The model, Preprocessing.pkl and Feature.pkl must come from the same notebook run.")
    if "prediction" in st.session_state:
        show_prediction(st.session_state["prediction"])


# ============================================================
# PAGE: MODEL INTELLIGENCE
# ============================================================
def render_comparison():
    st.markdown("### Model comparison")
    st.markdown(
        '<div class="panel">Several machine learning and deep learning algorithms were trained on the same preprocessed data and '
        'evaluated on one shared held-out test set during development.</div>', unsafe_allow_html=True)
    st.write("")
    ranked = sorted(NOTEBOOK_COMPARISON.items(), key=lambda kv: kv[1])  # ascending → best bar on top
    fig = go.Figure()
    for family, color in [("Machine Learning", CYAN), ("Deep Learning", GOLD)]:
        items = [(n, v) for n, v in ranked if (n in DEEP_LEARNING_MODELS) == (family == "Deep Learning")]
        fig.add_trace(go.Bar(
            x=[v * 100 for _, v in items], y=[n for n, _ in items], orientation="h", name=family,
            marker=dict(color=color), text=[f"{v * 100:.2f}%" for _, v in items], textposition="outside", cliponaxis=False,
            textfont=dict(color=TEXT), hovertemplate="%{y}<br>Test accuracy: %{x:.2f}%<extra>" + family + "</extra>"))
    fig.update_layout(barmode="overlay")
    fig.update_yaxes(categoryorder="array", categoryarray=[n for n, _ in ranked])
    fig.update_xaxes(range=[90, 100], title="Test Accuracy (%)")
    show_chart(style_fig(fig, 400, title="Test accuracy by algorithm"))
    # Final model selection: reported honestly from the numbers
    best_name, best_val = ranked[-1]
    ann_val = NOTEBOOK_COMPARISON["Artificial Neural Network"]
    ann_rank = [n for n, _ in reversed(ranked)].index("Artificial Neural Network") + 1
    gap_pp = (best_val - ann_val) * 100
    gap_samples = round((best_val - ann_val) * NOTEBOOK_ANN["test_samples"])
    runner_up = ranked[-2]
    st.markdown("### Why the ANN is chosen as the final model")
    st.markdown(
        '<div class="panel" style="margin-top:12px"><b>the ANN learns how the inputs work together.</b><br>'
        '<span class="muted">Weather type is rarely decided by a single measurement. The same reading can mean different things depending on '
        'the others: for example, a high precipitation value alongside a very low temperature points towards snow, while the same '
        'precipitation at a warm temperature points towards rain. A neural network is designed to discover these relationships and '
        'interactions among the inputs by itself, directly from the training data.</span></div>'
        '<div class="reason-grid">'
        '<div class="card"><div class="kicker">01</div><h4>Finds relationships among inputs</h4>'
        '<p>Every neuron in the first hidden layer weighs all 11 encoded inputs at the same time. The next layers 32, 16 and 8 neurons '
        'combine those results into higher-level patterns.</p></div>'
        '<div class="card"><div class="kicker">02</div><h4>Captures non-linear patterns</h4>'
        '<p>ReLU activations let the network draw curved boundaries between weather types instead of straight lines.</p></div>'
        '<div class="card"><div class="kicker">03</div><h4>Gives a probability for every class</h4>'
        '<p>The Softmax output layer returns a probability for Sunny, Cloudy, Rainy and Snowy.</p></div>'
        '<div class="card"><div class="kicker">04</div><h4>Controlled against overfitting</h4>'
        '<p>Batch Normalization, Dropout, L2 regularization and early stopping keep training stable. '
        f'Scores stay close across the splits: train {NOTEBOOK_ANN["train_accuracy"]:.2%}, validation {NOTEBOOK_ANN["val_accuracy"]:.2%}, '
        f'test {NOTEBOOK_ANN["test_accuracy"]:.2%}.</p></div>'
        '</div>',
        unsafe_allow_html=True)


def render_architecture():
    st.markdown("### ANN architecture")
    layers = [("Input", "11 encoded features"), ("Dense", "64 neurons · ReLU"), ("Dense", "32 neurons · ReLU"),
              ("Dense", "16 neurons · ReLU"), ("Dense", "8 neurons · ReLU"), ("Output", "4 classes · Softmax")]
    for col, (name, detail) in zip(st.columns(6), layers):
        with col:
            st.markdown(f'<div class="card" style="min-height:90px"><div class="kicker">{name}</div><div style="font-family:Plus Jakarta Sans,Segoe UI,sans-serif;font-weight:700;margin-top:4px">{detail}</div></div>',
                        unsafe_allow_html=True)
    params = f" The loaded model has {model.count_params():,} parameters." if model is not None else ""

def render_training_curves():
    st.markdown("### ANN training curves")
    h = TRAINING_HISTORY
    epochs = list(range(1, len(h["loss"]) + 1))
    best = NOTEBOOK_ANN["best_epoch"]
    c1, c2 = st.columns(2)
    for col, (key, title, ytitle) in zip((c1, c2), [("loss", "Loss", "Loss"), ("accuracy", "Accuracy", "Accuracy")]):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=h[key], name=f"Training {title.lower()}", mode="lines", line=dict(color=CYAN, width=2.5),
                                 hovertemplate="Epoch %{x}<br>%{y:.4f}<extra>Training</extra>"))
        fig.add_trace(go.Scatter(x=epochs, y=h[f"val_{key}"], name=f"Validation {title.lower()}", mode="lines", line=dict(color=GOLD, width=2.5),
                                 hovertemplate="Epoch %{x}<br>%{y:.4f}<extra>Validation</extra>"))
        fig.add_vline(x=best, line_dash="dot", line_color=VIOLET)
        # Label sits centred on the dotted line, just above the plot area, so it never covers the curves
        fig.add_annotation(x=best, y=1, xref="x", yref="paper", yanchor="bottom", xanchor="center", showarrow=False,
                           text=f"Best epoch {best}", font=dict(color=VIOLET, size=13),
                           bgcolor="rgba(15,29,56,.92)", bordercolor=VIOLET, borderwidth=1, borderpad=4)
        fig.update_xaxes(title="Epoch")
        fig.update_yaxes(title=ytitle)
        with col:
            show_chart(style_fig(fig, 360, title=f"Training vs validation {title.lower()}"))

def render_evaluation():
    st.markdown("### ANN performance evaluation")
    try:
        ev = compute_evaluation(model, preprocessor, data, tuple(classes), tuple(feature_cols), MODEL_PATH.stat().st_mtime)
    except Exception as exc:
        st.error(f"Could not reproduce evaluation: {exc}")
        st.info("Check that ann_model.keras, Preprocessing.pkl, LabelEncoding.pkl and Feature.pkl were all saved from the same notebook run.")
        return
    y_true, y_pred = ev["test"]["y"], ev["test"]["pred"]
    labels = list(range(len(classes)))
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    row1, row2 = st.columns(3), st.columns(3)
    cards = [("TRAINING ACCURACY", f'{ev["train"]["accuracy"]:.2%}', f'{ev["train"]["n"]:,} samples'),
             ("VALIDATION ACCURACY", f'{ev["val"]["accuracy"]:.2%}', f'{ev["val"]["n"]:,} samples'),
             ("TEST ACCURACY", f'{ev["test"]["accuracy"]:.2%}', f'{ev["test"]["n"]:,} samples'),
             ("WEIGHTED PRECISION", f"{precision:.2%}", "Test set"),
             ("WEIGHTED RECALL", f"{recall:.2%}", "Test set"),
             ("WEIGHTED F1-SCORE", f"{f1:.2%}", "Test set")]
    for col, (label, value, note) in zip(row1 + row2, cards):
        with col:
            metric_card(label, value, note)

    st.write("")
    left, right = st.columns([1, 1.1], gap="large")
    with left:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig = px.imshow(cm, x=classes, y=classes, text_auto=True, aspect="auto",
                        color_continuous_scale=["#0F1D38", "#1D5FA8", "#1B7FBF"],
                        labels=dict(x="Predicted class", y="Actual class", color="Samples"))
        fig.update_traces(textfont=dict(color=TEXT, size=15, family=FONT_HEAD), hovertemplate="Actual %{y}<br>Predicted %{x}<br>%{z} samples<extra></extra>")
        ticks = [f'<span style="color:{wcolor(n)}">{n}</span>' for n in classes]
        fig.update_xaxes(tickmode="array", tickvals=classes, ticktext=ticks, side="bottom", showgrid=False)
        fig.update_yaxes(tickmode="array", tickvals=classes, ticktext=ticks, autorange="reversed", showgrid=False)
        fig.update_coloraxes(showscale=False)
        show_chart(style_fig(fig, 420, title="Confusion matrix", legend=False))
    with right:
        st.markdown('<div class="sec-title">Classification report</div>', unsafe_allow_html=True)
        report = classification_report(y_true, y_pred, labels=labels, target_names=classes, output_dict=True, zero_division=0)
        render_report_table(report, classes)

def render_model():
    page_header("03 / MODEL TRANSPARENCY", "Model intelligence",
                "How the ANN compares with other algorithms, how it was trained and how it performs.")
    render_comparison()
    render_architecture()
    render_training_curves()
    if model is None:
        st.warning("Model files are missing or could not be loaded, so live evaluation is unavailable. See sidebar → Required files.")
        for problem in artifact_problems:
            st.error(problem)
    else:
        render_evaluation()


# ============================================================
# PAGE: WEATHER ANALYTICS
# ============================================================
def render_analytics():
    page_header("04 / EXPLORATORY WEATHER ANALYTICS", "Weather pattern analytics"
                )
    options = sorted(data[TARGET].unique())
    selected = st.multiselect("Weather classes", options=options, default=options)
    view = data[data[TARGET].isin(selected)]
    c1, c2, c3 = st.columns(3)
    has = len(view) > 0
    with c1: metric_card("FILTERED RECORDS", f"{len(view):,}", "Records in current selection")
    with c2: metric_card("AVERAGE TEMPERATURE", f'{view["Temperature (°C)"].mean():.1f} °C' if has else "—", "Across selected records" if has else "No selection")
    with c3: metric_card("AVERAGE HUMIDITY", f'{view["Humidity (%)"].mean():.1f}%' if has else "—", "Across selected records" if has else "No selection")
    if not has:
        st.info("Select at least one weather class to see the charts.")
        return
    st.write("")
    H = 400

    left, right = st.columns(2, gap="medium")
    with left:
        measures = ["Temperature (°C)", "Humidity (%)", "Wind Speed (km/h)", "Precipitation (%)", "Visibility (km)"]
        measure_colors = dict(zip(measures, ["#FB7185", "#34D399", CYAN, VIOLET, GOLD]))
        agg = view.groupby(TARGET)[measures].mean().reset_index()
        long = agg.melt(id_vars=TARGET, var_name="Measure", value_name="Average")
        fig = px.bar(long, x=TARGET, y="Average", color="Measure", barmode="group",
                     color_discrete_map=measure_colors, category_orders={TARGET: options, "Measure": measures})
        fig.update_traces(hovertemplate="%{x}<br>%{fullData.name}: %{y:.1f}<extra></extra>")
        fig.update_xaxes(title="")
        fig.update_yaxes(title="Average value")
        show_chart(style_fig(fig, H + 60, title="Weather characteristics comparison"))
        chart_note("Average temperature, humidity, wind speed, precipitation and visibility for each weather class.")
    with right:
        corr = view[NUMERIC_FEATURES].corr()
        fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, aspect="auto",
                        color_continuous_scale=["#7C5CD6", "#0F1D38", "#1B7FBF"])
        fig.update_traces(textfont=dict(color=TEXT, size=11), hovertemplate="%{y} vs %{x}<br>r = %{z:.2f}<extra></extra>")
        fig.update_xaxes(tickangle=-35, showgrid=False)
        fig.update_yaxes(showgrid=False, autorange="reversed")
        fig.update_coloraxes(colorbar=dict(title=dict(text="r", font=dict(color=TEXT)), tickfont=dict(color=MUTED), thickness=12))
        show_chart(style_fig(fig, H + 60, title="Feature correlation heatmap", legend=False))
        chart_note("correlation between numeric features.Violet = negative, blue = positive.")

    left, right = st.columns(2, gap="medium")
    with left:
        cloud = view[CLOUD_COL].str.title().value_counts().rename_axis("Cloud Cover").reset_index(name="Records")
        fig = px.pie(cloud, names="Cloud Cover", values="Records", hole=.58, color="Cloud Cover", color_discrete_map=CLOUD_COLORS)
        fig.update_traces(textposition="outside", texttemplate="%{label}<br>%{percent}<br>%{value:,} records",
                          textfont=dict(color=TEXT, size=11), marker=dict(line=dict(color=BG, width=2)),
                          hovertemplate="%{label}<br>%{value:,} records (%{percent})<extra></extra>")
        show_chart(style_fig(fig, H + 20, title="Cloud cover distribution", legend=False))
        
    with right:
        st.markdown('<div class="sec-title">Selection summary by weather class</div>', unsafe_allow_html=True)
        summary = view.groupby(TARGET).agg(
            Records=("Temperature (°C)", "size"), **{"Avg temp (°C)": ("Temperature (°C)", "mean"), "Avg humidity (%)": ("Humidity (%)", "mean"),
                                         "Avg pressure (hPa)": ("Atmospheric Pressure (hpa)", "mean"), "Avg UV index": ("UV Index", "mean")}).round(1)
        render_table(summary, index=True, index_label="Weather Type", max_height=330)
        chart_note("Quick numeric summary for the classes currently selected.")

    st.markdown("### Weather class distributions")
    st.caption("Boxplots of every numeric measurement by weather class: the box is the middle 50% of values, "
               "the line inside is the median and the whiskers show the typical range.")
    # One boxplot per numeric column (Cloud Cover is categorical, so it has its own donut chart above)
    for i in range(0, len(NUMERIC_FEATURES), 2):
        for slot, col in zip(st.columns(2, gap="medium"), NUMERIC_FEATURES[i:i + 2]):
            with slot:
                fig = px.box(view, x=TARGET, y=col, color=TARGET, color_discrete_map=ANALYTICS_WEATHER_COLORS,
                             points=False, category_orders={TARGET: options})
                fig.update_xaxes(title="")
                show_chart(style_fig(fig, 360, title=f"{META_BY_COL[col]['label']} by weather class", legend=False))


# ============================================================
# PAGE: DATASET EXPLORER
# ============================================================
def render_explorer():
    page_header("05 / DATA INSPECTION", "Dataset explorer")
    c1, c2, c3 = st.columns(3)
    with c1: metric_card("ROWS", f"{len(data):,}", "Original records")
    with c2: metric_card("COLUMNS", f"{data.shape[1]}", "Including target")
    with c3: metric_card("MISSING VALUES", f"{int(data.isna().sum().sum())}", "Across all columns")
    st.markdown("### Preview records")
    chosen = st.multiselect("Choose columns", data.columns.tolist(), default=data.columns.tolist())
    if chosen:
        render_table(data[chosen].head(100), max_height=420)
        st.caption("Showing the first 100 records.")
    else:
        st.info("Choose at least one column to preview.")
    st.markdown("### Descriptive statistics")
    render_table(data.describe(include="all").T, index=True, index_label="Column", max_height=420)
    st.markdown("### Data quality")
    quality = pd.DataFrame({"Data type": data.dtypes.astype(str), "Missing values": data.isna().sum(), "Unique values": data.nunique()})
    render_table(quality, index=True, index_label="Column", max_height=420)


# ============================================================
# PAGE: ABOUT
# ============================================================
def render_about():
    page_header("06 / PROJECT DOCUMENTATION", "About this project")
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        badges = "".join(
            f'<span class="badge" style="color:{c};border-color:{c}66;background:{c}1A">{WEATHER_ICONS[n]} {n}</span>'
            for n, c in WEATHER_COLORS.items())
        st.markdown(
            '<div class="panel"><h3>About the Project</h3>'
            '<p>Weather classification assigns an observation to a weather category from measured atmospheric conditions. '
            'This project does that with an Artificial Neural Network trained on a labelled weather dataset.</p>'
            '<p>The model reads eight inputs: temperature, humidity, wind speed, precipitation, cloud cover, atmospheric pressure, '
            'UV index and visibility. It converts them into a probability for each of four categories and reports the most likely one.</p>'
            f'<div style="margin-top:12px">{badges}</div></div>', unsafe_allow_html=True)
        st.write("")
        rows = [("Task", "Weather Type Classification"), ("Framework", "TensorFlow / Keras"), ("Model", "Artificial Neural Network"),
                ("Preprocessing", "StandardScaler + OneHotEncoder"), ("Input Features", "8"), ("Output Classes", "4")]
        kv = "".join(f'<div class="k">{k}</div><div class="v">{v}</div>' for k, v in rows)
        st.markdown(f'<div class="panel"><h3>Technical Summary</h3><div class="kv">{kv}</div></div>', unsafe_allow_html=True)
    with right:
        steps = [("Enter Conditions", "You select the weather measurements."),
                 ("Validate Input", "Slider ranges keep every value within the range seen in the training data."),
                 ("Encode Categories", "Cloud Cover is one-hot encoded by the saved preprocessing pipeline."),
                 ("Scale Numerical Features", "The seven numeric measurements are standardized with the fitted scaler."),
                 ("Run the Neural Network", "The ANN processes the 11 transformed features."),
                 ("Display Result", "The predicted class, confidence and class probabilities are shown.")]
        items = "".join(f'<div class="step"><div class="n">{i}</div><div><b>{t}</b><span>{d}</span></div></div>'
                        for i, (t, d) in enumerate(steps, 1))
        st.markdown(f'<div class="sec-title">Prediction Pipeline</div><div class="pipe">{items}</div>', unsafe_allow_html=True)

    st.markdown("### Input features")
    info = pd.DataFrame({
        "Feature": FEATURES,
        "Description": ["Air temperature in degrees Celsius", "Relative humidity percentage", "Wind speed in km/h",
                        "Precipitation percentage", "Cloud cover category (clear, partly cloudy, cloudy, overcast)",
                        "Atmospheric pressure in hPa", "Ultraviolet index", "Visibility distance in kilometres"]})
    render_table(info, max_height=420)
    st.markdown("### Important model note")
    st.warning(
            "⚠️  This prediction is generated by a deep learning model "
                    "and may not always be correct. The confidence score is model-estimated and "
                    "does not guarantee that the prediction is accurate.")
    st.info(
            "💡 **Tip :** If the prediction doesn't match the expected weather type, "
                    "try different input values and make 2–3 predictions to compare the results.")
    st.warning(
            "⚠️  This project is developed for educational and demonstration purposes. It is not a real-time weather forecasting system."
        )


# ============================================================
# ROUTER
# ============================================================
{
    "Overview": render_overview,
    "Predict Weather": render_predict,
    "Model Intelligence": render_model,
    "Weather Analytics": render_analytics,
    "Dataset Explorer": render_explorer,
    "About": render_about,
}[page]()
render_footer()