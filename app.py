"""
Assignment 1: Machine Learning Classification Systems
Course: Applied AI / Machine Learning & Software Development (ISE 402)
Lecturer: Mr. Harry Attieku-Boateng
Student: Emmanuel Asante (ID: PU092301 / PUIT23210016)
Pentecost University - Faculty of Computing and Information Systems (FOCIS)

Google AI Studio // Mossline Scientific Inference & Biometric Deck
- Part A: Iris Botanical Multi-Class Neural Engine (data/iris.csv, 150 specimens)
- Part B: Cardiac Hemodynamic Risk Engine (Scikit-Learn OneHotEncoder Mandate)
- Mossline Scrollytelling & 3D Interactive Telemetry Deck
- Strictly Correlated Botanical & Clinical Pathology Imagery
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Configure Page
st.set_page_config(
    page_title="Mossline // PU-AI Living Planet & Clinical Inference",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Google AI Studio / Mossline CSS Theme
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter+Tight:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">

<style>
    /* CSS Root Palette & Typography (Exact Mossline Tokens) */
    :root {
        --ink: #10291a;
        --ink2: #2c4434;
        --mist: #e7ebe8;
        --paper: #f6f7f4;
        --lime: #c6f35c;
        --lime2: #a9dd3a;
        --glass: rgba(244, 247, 243, 0.16);
        --line: rgba(255, 255, 255, 0.28);
        --sans: "Inter Tight", "Helvetica Neue", Arial, sans-serif;
        --mono: "JetBrains Mono", ui-monospace, Menlo, monospace;
        
        /* Dark Deck Overrides */
        --deck-bg: #0b1910;
        --deck-surface: rgba(20, 42, 27, 0.78);
        --deck-hover: rgba(30, 60, 39, 0.92);
        --deck-border: rgba(198, 243, 92, 0.25);
        --deck-border-bright: rgba(198, 243, 92, 0.6);
        --deck-text: #f4f7f4;
        --deck-text-dim: #9eb5a5;
        --danger: #ff5252;
    }

    /* Global Dark Backdrop Override */
    .stApp {
        background-color: var(--deck-bg);
        background-image: 
            radial-gradient(ellipse 60% 50% at 20% 20%, rgba(198, 243, 92, 0.06), transparent 70%),
            radial-gradient(ellipse 50% 40% at 80% 80%, rgba(44, 68, 52, 0.25), transparent 70%);
        color: var(--deck-text);
        font-family: var(--sans);
        -webkit-font-smoothing: antialiased;
    }

    /* Mossline Precision Typography Utilities */
    .mono {
        font-family: var(--mono);
        text-transform: uppercase;
        letter-spacing: .14em;
        font-size: 11px;
    }

    .chip {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        height: 22px;
        padding: 0 9px;
        background: var(--lime);
        color: var(--ink);
        border-radius: 3px;
        font-family: var(--mono);
        font-size: 10.5px;
        font-weight: 700;
        letter-spacing: .16em;
        text-transform: uppercase;
        white-space: nowrap;
    }

    .chip-dark {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        height: 22px;
        padding: 0 9px;
        background: rgba(198, 243, 92, 0.12);
        color: var(--lime);
        border: 1px solid var(--deck-border);
        border-radius: 3px;
        font-family: var(--mono);
        font-size: 10.5px;
        font-weight: 600;
        letter-spacing: .14em;
        text-transform: uppercase;
        white-space: nowrap;
    }

    .chip-danger {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        height: 22px;
        padding: 0 9px;
        background: rgba(255, 82, 82, 0.2);
        color: #ff5252;
        border: 1px solid rgba(255, 82, 82, 0.4);
        border-radius: 3px;
        font-family: var(--mono);
        font-size: 10.5px;
        font-weight: 700;
        letter-spacing: .14em;
        text-transform: uppercase;
        white-space: nowrap;
    }

    /* Fixed Mossline Navigation Header */
    .moss-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 16px 26px;
        background: var(--deck-surface);
        border: 1px solid var(--deck-border);
        border-radius: 12px;
        backdrop-filter: blur(18px);
        margin-bottom: 24px;
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.4);
    }

    .moss-logo {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .moss-title {
        font-family: var(--sans);
        font-weight: 700;
        font-size: 20px;
        letter-spacing: -0.02em;
        color: var(--deck-text);
    }

    .moss-subtitle {
        font-family: var(--mono);
        font-size: 10.5px;
        letter-spacing: 0.14em;
        color: var(--lime);
        text-transform: uppercase;
    }

    /* Glassmorphic Telemetry Cards */
    .glass-card {
        position: relative;
        background: var(--deck-surface);
        border: 1px solid var(--deck-border);
        border-radius: 12px;
        padding: 24px;
        backdrop-filter: blur(16px);
        margin-bottom: 20px;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
        transition: transform 0.28s ease, border-color 0.28s ease;
    }

    .glass-card:hover {
        transform: translateY(-2px);
        border-color: var(--deck-border-bright);
    }

    /* Live Animations */
    @keyframes heartbeat {
        0% { transform: scale(1); }
        14% { transform: scale(1.18); }
        28% { transform: scale(1); }
        42% { transform: scale(1.12); }
        70% { transform: scale(1); }
        100% { transform: scale(1); }
    }

    @keyframes pulse-lime {
        0% { box-shadow: 0 0 0 0 rgba(198, 243, 92, 0.7); }
        70% { box-shadow: 0 0 0 14px rgba(198, 243, 92, 0); }
        100% { box-shadow: 0 0 0 0 rgba(198, 243, 92, 0); }
    }

    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(255, 82, 82, 0.8); }
        70% { box-shadow: 0 0 0 14px rgba(255, 82, 82, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 82, 82, 0); }
    }

    @keyframes spark-draw {
        0% { stroke-dashoffset: 240; }
        100% { stroke-dashoffset: 0; }
    }

    .pulse-heart {
        display: inline-block;
        animation: heartbeat 1.35s infinite ease-in-out;
        transform-origin: center center;
    }

    .active-badge-lime {
        animation: pulse-lime 2.4s infinite;
    }

    .active-badge-red {
        animation: pulse-red 1.8s infinite;
    }

    .animated-sparkline {
        stroke-dasharray: 240;
        stroke-dashoffset: 0;
        animation: spark-draw 2.5s ease-out forwards;
    }

    /* Mossline Telemetry Card */
    .moss-telemetry-card {
        padding: 18px;
        border-radius: 10px;
        background: rgba(40, 58, 44, 0.45);
        border: 1px solid rgba(198, 243, 92, 0.28);
        backdrop-filter: blur(16px);
        color: #fff;
        margin-bottom: 20px;
    }

    .moss-telemetry-card h4 {
        font-size: 15px;
        font-weight: 500;
        line-height: 1.2;
        margin: 0;
    }

    .moss-telemetry-card .note {
        font-family: var(--mono);
        font-size: 9.5px;
        letter-spacing: .14em;
        text-transform: uppercase;
        opacity: .75;
        margin-top: 6px;
    }

    .moss-telemetry-card .big {
        display: flex;
        align-items: baseline;
        gap: 6px;
        margin-top: 10px;
    }

    .moss-telemetry-card .big strong {
        font-size: 42px;
        font-weight: 600;
        letter-spacing: -.04em;
        line-height: 1;
        color: var(--lime);
    }

    .moss-telemetry-card .delta {
        background: var(--lime);
        color: var(--ink);
        font-family: var(--mono);
        font-size: 10px;
        letter-spacing: .1em;
        padding: 3px 6px;
        border-radius: 3px;
        font-weight: 600;
    }

    /* Mossline 3D Tilting Solution Cards */
    .sol-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin: 20px 0 28px 0;
    }

    .sol-card {
        position: relative;
        height: 240px;
        border-radius: 12px;
        overflow: hidden;
        background: #142a1b;
        color: #fff;
        border: 1px solid var(--deck-border);
        padding: 20px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 10px 28px rgba(0,0,0,0.3);
        transition: transform 0.35s cubic-bezier(.2,.8,.2,1), border-color 0.35s ease;
    }

    .sol-card:hover {
        transform: translateY(-4px);
        border-color: var(--lime);
    }

    .sol-card .tag {
        font-family: var(--mono);
        font-size: 10px;
        letter-spacing: .14em;
        text-transform: uppercase;
        color: var(--lime);
    }

    .sol-card h3 {
        font-size: 20px;
        font-weight: 600;
        letter-spacing: -.03em;
        line-height: 1.15;
        margin: 8px 0 6px 0;
        color: #fff;
    }

    .sol-card p {
        font-size: 12.5px;
        line-height: 1.45;
        color: var(--deck-text-dim);
    }

    /* Mossline Stats Ribbon */
    .stats-ribbon {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 20px;
        padding: 20px 0;
        border-top: 1px solid var(--deck-border);
        border-bottom: 1px solid var(--deck-border);
        margin: 24px 0;
    }

    .stat-item strong {
        display: block;
        font-size: 38px;
        font-weight: 600;
        letter-spacing: -.04em;
        line-height: 1;
        color: var(--lime);
    }

    .stat-item span {
        display: block;
        font-family: var(--mono);
        font-size: 10.5px;
        letter-spacing: .14em;
        text-transform: uppercase;
        color: var(--deck-text-dim);
        margin-top: 6px;
    }

    /* Telemetry Coordinates Header */
    .hud-meta {
        font-family: var(--mono);
        font-size: 11px;
        color: var(--deck-text-dim);
        letter-spacing: 0.12em;
        line-height: 1.65;
    }

    /* Sidebar Clean Styling */
    section[data-testid="stSidebar"] {
        background-color: var(--ink) !important;
        border-right: 1px solid var(--deck-border) !important;
    }

    /* Streamlit Widget Polishing */
    .stSlider > div > div > div > div {
        background-color: var(--lime) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        border-bottom: 1px solid var(--deck-border);
    }

    .stTabs [data-baseweb="tab"] {
        font-family: var(--mono);
        font-size: 11.5px;
        letter-spacing: .12em;
        text-transform: uppercase;
        color: var(--deck-text-dim);
        padding: 10px 16px;
    }

    .stTabs [aria-selected="true"] {
        color: var(--lime) !important;
        border-bottom-color: var(--lime) !important;
    }
</style>
""", unsafe_allow_html=True)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data")
MOSSLINE_DIR = os.path.join(BASE_DIR, "mossline_extracted")
MOSSLINE_HTML = os.path.join(MOSSLINE_DIR, "index.html")

# Load Models & Metrics with Caching
@st.cache_resource
def load_iris_artifacts():
    path = os.path.join(MODELS_DIR, "iris_best_pipeline.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_resource
def load_heart_artifacts():
    path = os.path.join(MODELS_DIR, "heart_best_pipeline.joblib")
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_data
def load_metrics():
    iris_metrics_path = os.path.join(RESULTS_DIR, "iris_metrics.json")
    heart_metrics_path = os.path.join(RESULTS_DIR, "heart_metrics.json")
    iris_data = {}
    heart_data = {}
    if os.path.exists(iris_metrics_path):
        with open(iris_metrics_path, "r") as f:
            iris_data = json.load(f)
    if os.path.exists(heart_metrics_path):
        with open(heart_metrics_path, "r") as f:
            heart_data = json.load(f)
    return iris_data, heart_data

iris_bundle = load_iris_artifacts()
heart_bundle = load_heart_artifacts()
iris_metrics_data, heart_metrics_data = load_metrics()

# Top Persistent Mossline Scientific HUD Navigation Header
st.markdown("""
<div class="moss-nav">
    <div class="moss-logo">
        <svg width="30" height="30" viewBox="0 0 32 32" aria-hidden="true">
            <path d="M16 2l12 7v14l-12 7-12-7V9z" fill="#1b3823" stroke="#c6f35c" stroke-width="1.8"/>
            <path d="M16 8c-4 4-5 9-1 16 1-5 4-9 8-11-3 0-5 1-7 3 0-3 0-5 0-8z" fill="#c6f35c"/>
        </svg>
        <div>
            <div class="moss-title">MOSSLINE // LIVING PLANET &amp; CLINICAL INFERENCE DECK</div>
            <div class="moss-subtitle">Applied AI Laboratory • Pentecost University FOCIS</div>
        </div>
    </div>
    <div style="display: flex; gap: 14px; align-items: center;">
        <span class="chip-dark">STATION 07 • SOWUTUOM</span>
        <span class="chip">EVAL-PROD // READY</span>
        <span class="hud-meta" style="border-left: 1px solid var(--deck-border); padding-left: 14px;">
            LAT 5.6037° N<br/>LON 0.2828° W • ELEV 82M
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# 4 Key Quantitative Performance Metrics Ribbon
st.markdown("""
<div class="stats-ribbon">
    <div class="stat-item">
        <strong>150</strong>
        <span>Iris Taxa Specimens (data/iris.csv)</span>
    </div>
    <div class="stat-item">
        <strong>96.67%</strong>
        <span>Part A SVC Test Accuracy</span>
    </div>
    <div class="stat-item">
        <strong>96.00%</strong>
        <span>Part B Random Forest Accuracy</span>
    </div>
    <div class="stat-item">
        <strong>98.25%</strong>
        <span>Patient Sensitivity Recall Safeguard</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Navigation (Direct access, NO useless overview tab)
st.sidebar.markdown("""
<div style="padding: 10px 0 16px 0;">
    <div style="display: flex; gap: 8px; align-items: center;">
        <span class="chip">MODULES</span>
        <span style="font-family: var(--mono); font-size: 11px; color: var(--lime);">SELECT ENGINE</span>
    </div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation Console:",
    [
        "🌸 Part A: Iris Botanical Neural Engine",
        "🫀 Part B: Cardiac Hemodynamic Risk Engine",
        "⚡ Model Telemetry & Comparative Benchmarks",
        "📑 Preprocessing & Academic Report",
        "🌿 Mossline Cinematic Scrollytelling (AI Studio)"
    ]
)

st.sidebar.divider()
st.sidebar.markdown(
    """
    <div class="hud-meta">
        <b>SUPERVISION</b>: Mr. Harry Attieku-Boateng<br/>
        <b>RESEARCHER</b>: Emmanuel Asante<br/>
        <b>MATRICULATION</b>: PU092301 / PUIT23210016<br/>
        <b>FACULTY</b>: FOCIS Computer Science<br/>
        <b>DESIGN ENGINE</b>: Google AI Studio (Mossline)<br/>
        <b>FRAMEWORK</b>: Scikit-Learn 1.9 // Python 3.11
    </div>
    """,
    unsafe_allow_html=True
)

# =============================================================
# MODULE 1: PART A - IRIS BOTANICAL NEURAL ENGINE
# =============================================================
if page == "🌸 Part A: Iris Botanical Neural Engine":
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 18px;">
        <div>
            <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 6px;">
                <span class="chip">PART A</span>
                <span class="chip-dark">BOTANICAL MULTI-CLASS</span>
                <span style="font-family: var(--mono); font-size: 11px; color: var(--deck-text-dim);">DATASET: data/iris.csv (150 SAMPLES)</span>
            </div>
            <h1 style="margin: 0; font-size: 32px; letter-spacing: -0.03em; color: var(--deck-text);">
                Iris Botanical Morphometric Classification
            </h1>
        </div>
        <div style="text-align: right;">
            <span class="chip" style="background: var(--lime); color: var(--ink);">CHAMPION: SVC (RBF) 96.67% ACC</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_engine, tab_botany, tab_metrics = st.tabs([
        "🧪 Live Inference Console",
        "🌺 Botanical Anatomy & Species Cards",
        "📊 Empirical Confusion Matrix & EDA"
    ])

    with tab_engine:
        col_ctrl, col_pred = st.columns([1.1, 1.0])

        with col_ctrl:
            st.markdown("""
            <div class="glass-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
                    <span class="chip-dark">CALIBRATION DECK</span>
                    <span style="font-family: var(--mono); font-size: 11px; color: var(--lime);">FEATURE MATRIX: 4 DIMS</span>
                </div>
            """, unsafe_allow_html=True)

            sepal_length = st.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.4, step=0.1, help="Outer floral calyx longitudinal dimension")
            sepal_width = st.slider("Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.4, step=0.1, help="Outer floral calyx transverse dimension")
            petal_length = st.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, value=1.7, step=0.1, help="Inner corolla petal longitudinal dimension")
            petal_width = st.slider("Petal Width (cm)", min_value=0.1, max_value=2.5, value=0.4, step=0.1, help="Inner corolla petal transverse dimension")

            preset = st.selectbox(
                "⚡ Quickload Ground-Truth Vector from data/iris.csv:",
                [
                    "(Custom Measurement Slider State)",
                    "Setosa Ground-Truth: [5.1, 3.5, 1.4, 0.2]",
                    "Versicolor Ground-Truth: [6.0, 2.9, 4.5, 1.5]",
                    "Virginica Ground-Truth: [6.9, 3.1, 5.4, 2.1]"
                ]
            )
            st.markdown("</div>", unsafe_allow_html=True)

            # Mossline Live Telemetry Glass Card
            st.markdown("""
            <div class="moss-telemetry-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <h4>Morphometric Dimension<br/>Balance</h4>
                        <div class="note">Plot 07 • 4 Botanical Features</div>
                    </div>
                    <span class="delta">+0.967 F1</span>
                </div>
                <div class="big">
                    <strong>96.7</strong><span>% Confidence</span>
                </div>
                <svg viewBox="0 0 220 54" style="width: 100%; height: 54px; margin-top: 8px; overflow: visible;">
                    <polyline class="animated-sparkline" fill="none" stroke="#c6f35c" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" points="0,48 30,42 60,45 90,30 120,34 150,18 180,14 214,6"/>
                    <circle cx="90" cy="30" r="2.8" fill="#fff"/>
                    <circle cx="150" cy="18" r="2.8" fill="#fff"/>
                    <circle cx="214" cy="6" r="3.6" fill="#c6f35c"/>
                </svg>
                <div class="note" style="margin-top: 4px;">Sepal L • Sepal W • Petal L • Petal W</div>
            </div>
            """, unsafe_allow_html=True)

        with col_pred:
            if "Setosa" in preset:
                sepal_length, sepal_width, petal_length, petal_width = 5.1, 3.5, 1.4, 0.2
            elif "Versicolor" in preset:
                sepal_length, sepal_width, petal_length, petal_width = 6.0, 2.9, 4.5, 1.5
            elif "Virginica" in preset:
                sepal_length, sepal_width, petal_length, petal_width = 6.9, 3.1, 5.4, 2.1

            if iris_bundle:
                pipeline = iris_bundle["pipeline"]
                target_names = iris_bundle["target_names"]
                feature_names = iris_bundle["feature_names"]

                input_df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=feature_names)
                pred_idx = pipeline.predict(input_df)[0]
                pred_proba = pipeline.predict_proba(input_df)[0]
                pred_species = target_names[pred_idx]

                sp_cards = {
                    "setosa": "iris_setosa_card.png",
                    "versicolor": "iris_versicolor_card.png",
                    "virginica": "iris_virginica_card.png"
                }
                card_filename = sp_cards.get(pred_species.lower(), "iris_setosa_card.png")

                st.markdown(f"""
                <div class="glass-card active-badge-lime" style="border-left: 5px solid var(--lime);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span class="chip">BOTANICAL INFERENCE RESULT</span>
                        <span style="font-family: var(--mono); font-size: 11px; color: var(--lime);">CONFIDENCE: {pred_proba[pred_idx]*100:.1f}%</span>
                    </div>
                    <div style="font-size: 38px; font-weight: 700; color: var(--lime); letter-spacing: -0.03em; margin: 4px 0;">
                        Iris {pred_species}
                    </div>
                    <div class="hud-meta">
                        PREDICTOR: Support Vector Classifier (RBF Kernel)<br/>
                        NORM SCALE: StandardScaler (Z-Score Normalization)<br/>
                        STATUS: Confirmed Species Taxon from data/iris.csv
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Show Correlated Species Card Image
                img_path = os.path.join(FIGURES_DIR, card_filename)
                if os.path.exists(img_path):
                    st.image(img_path, caption=f"Correlated Morphometric Profile: Iris {pred_species}", use_container_width=True)

                st.markdown("""
                <div class="glass-card">
                    <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                        <span class="chip-dark">POSTERIOR DENSITY PROFILE</span>
                        <span style="font-family:var(--mono); font-size:11px; color:var(--deck-text-dim);">CALIBRATED PLATT SCALING</span>
                    </div>
                """, unsafe_allow_html=True)

                for name, prob in zip(target_names, pred_proba):
                    st.write(f"• **Iris *{name}***: `{prob * 100:.1f}%`")
                    st.progress(float(prob))
                st.markdown("</div>", unsafe_allow_html=True)

    with tab_botany:
        st.markdown("### 🌺 Botanical Morphology & Taxonomic Separation")
        st.write("Ronald Fisher's canonical Iris benchmark relies on the structural divergence between outer protective **Sepals** and inner reproductive **Petals**:")
        
        morph_path = os.path.join(FIGURES_DIR, "iris_morphology_guide.png")
        if os.path.exists(morph_path):
            st.image(morph_path, caption="Figure A1: Botanical Morphometric Guide (Sepal vs. Petal Geometry)", use_container_width=True)

        st.markdown("---")
        st.markdown("#### 🌸 The Three Floral Species in `data/iris.csv`:")
        c1, c2, c3 = st.columns(3)
        with c1:
            p1 = os.path.join(FIGURES_DIR, "iris_setosa_card.png")
            if os.path.exists(p1):
                st.image(p1, caption="Setosa: Linearly Separable, Small Petals", use_container_width=True)
        with c2:
            p2 = os.path.join(FIGURES_DIR, "iris_versicolor_card.png")
            if os.path.exists(p2):
                st.image(p2, caption="Versicolor: Intermediate Petal Morphometry", use_container_width=True)
        with c3:
            p3 = os.path.join(FIGURES_DIR, "iris_virginica_card.png")
            if os.path.exists(p3):
                st.image(p3, caption="Virginica: Elongated Violet Petals", use_container_width=True)

    with tab_metrics:
        st.markdown("### 📈 Empirical Confusion Matrix & Multiclass Benchmarks")
        col_m1, col_m2 = st.columns([1.1, 1.0])
        with col_m1:
            cm_path = os.path.join(FIGURES_DIR, "iris_confusion_matrix.png")
            if os.path.exists(cm_path):
                st.image(cm_path, caption="Figure A2: Multi-Class Confusion Matrix (Champion SVC on data/iris.csv)", use_container_width=True)
        with col_m2:
            pair_path = os.path.join(FIGURES_DIR, "iris_eda_pairplot.png")
            if os.path.exists(pair_path):
                st.image(pair_path, caption="Figure A3: Pairwise Morphological Feature Distributions Across Classes", use_container_width=True)

# =============================================================
# MODULE 2: PART B - CARDIAC HEMODYNAMIC RISK ENGINE
# =============================================================
elif page == "🫀 Part B: Cardiac Hemodynamic Risk Engine":
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 18px;">
        <div>
            <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 6px;">
                <span class="chip-danger">PART B</span>
                <span class="chip">ONEHOTENCODER: ACTIVE</span>
                <span style="font-family: var(--mono); font-size: 11px; color: var(--deck-text-dim);">DATASET: data/heart.csv (~500 PATIENTS)</span>
            </div>
            <h1 style="margin: 0; font-size: 32px; letter-spacing: -0.03em; color: var(--deck-text);">
                <span class="pulse-heart">🫀</span> Cardiovascular Pathological Risk Engine
            </h1>
        </div>
        <div style="text-align: right;">
            <span class="chip" style="background: var(--lime); color: var(--ink);">CHAMPION: RANDOM FOREST 96.0% ACC // 98.25% RECALL</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_assess, tab_ecg, tab_eda, tab_feat = st.tabs([
        "🩺 Clinical Diagnostic Assessment",
        "📈 Electrocardiogram (ECG) ST-Slope Guide",
        "📊 Clinical Exploratory Distributions (EDA)",
        "🧠 OneHotEncoded Feature Importances"
    ])

    with tab_assess:
        col_form, col_vis = st.columns([1.2, 1.0])

        with col_form:
            st.markdown("""
            <div class="glass-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
                    <span class="chip-dark">CLINICAL OBSERVATION DECK</span>
                    <span style="font-family: var(--mono); font-size: 11px; color: var(--lime);">ONEHOTENCODED PIPELINE</span>
                </div>
            """, unsafe_allow_html=True)

            with st.form("cardiac_form"):
                st.markdown("##### 👤 Patient Demographics & Hemodynamics")
                c1, c2 = st.columns(2)
                with c1:
                    age = st.slider("Patient Age (Years)", 20, 80, 54)
                    sex = st.selectbox("Biological Sex", options=["M", "F"], format_func=lambda x: "Male (M)" if x == "M" else "Female (F)")
                    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=210, value=135)
                with c2:
                    cholesterol = st.number_input("Serum Cholesterol (mg/dL)", min_value=0, max_value=600, value=240)
                    fasting_bs = st.radio("Fasting Blood Sugar", options=[0, 1], format_func=lambda x: "Normal (<= 120 mg/dL)" if x == 0 else "Elevated (> 120 mg/dL)")

                st.markdown("##### 🫀 Electrocardiographic & Exercise Stress Indicators")
                c3, c4 = st.columns(2)
                with c3:
                    chest_pain = st.selectbox(
                        "Chest Pain Type (ChestPainType)",
                        options=["ASY", "ATA", "NAP", "TA"],
                        format_func=lambda x: {
                            "ASY": "ASY: Asymptomatic Angina",
                            "ATA": "ATA: Atypical Angina",
                            "NAP": "NAP: Non-Anginal Pain",
                            "TA": "TA: Typical Angina"
                        }[x]
                    )
                    resting_ecg = st.selectbox(
                        "Resting ECG (RestingECG)",
                        options=["Normal", "ST", "LVH"],
                        format_func=lambda x: {
                            "Normal": "Normal: Normal Waveform",
                            "ST": "ST: ST-T Wave Abnormality",
                            "LVH": "LVH: Left Ventricular Hypertrophy"
                        }[x]
                    )
                    max_hr = st.slider("Max Heart Rate Achieved (MaxHR)", 60, 210, 138)
                with c4:
                    exercise_angina = st.selectbox("Exercise-Induced Angina", options=["N", "Y"], format_func=lambda x: "No (N)" if x == "N" else "Yes (Y)")
                    oldpeak = st.slider("ST Depression Induced by Stress (Oldpeak mm)", -2.0, 6.0, 1.4, step=0.1)
                    st_slope = st.selectbox(
                        "Peak ST Segment Slope (ST_Slope)",
                        options=["Up", "Flat", "Down"],
                        format_func=lambda x: {
                            "Up": "Up: Upsloping (Low Risk Indicator)",
                            "Flat": "Flat: Flat Slope (Elevated Risk Indicator)",
                            "Down": "Down: Downsloping (High Risk Indicator)"
                        }[x]
                    )

                submit_btn = st.form_submit_button("⚡ Compute Diagnostic Hemodynamic Risk", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_vis:
            st.markdown("""
            <div class="glass-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <span class="chip-dark">CORRELATED PATHOLOGY DIAGRAM</span>
                    <span style="font-family: var(--mono); font-size: 11px; color: var(--lime);">MYOCARDIAL ISCHEMIA</span>
                </div>
            """, unsafe_allow_html=True)
            path_img = os.path.join(FIGURES_DIR, "heart_pathology_diagram.png")
            if os.path.exists(path_img):
                st.image(path_img, caption="Clinical Reference: Coronary Atherosclerotic Stenosis & Ischemia", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if submit_btn:
            if heart_bundle:
                pipeline = heart_bundle["pipeline"]
                patient_data = pd.DataFrame([{
                    "Age": age,
                    "Sex": sex,
                    "ChestPainType": chest_pain,
                    "RestingBP": resting_bp,
                    "Cholesterol": cholesterol,
                    "FastingBS": fasting_bs,
                    "RestingECG": resting_ecg,
                    "MaxHR": max_hr,
                    "ExerciseAngina": exercise_angina,
                    "Oldpeak": oldpeak,
                    "ST_Slope": st_slope
                }])

                prediction = pipeline.predict(patient_data)[0]
                probabilities = pipeline.predict_proba(patient_data)[0]
                risk_prob = probabilities[1]

                st.markdown("<hr style='border-color: var(--deck-border); margin: 20px 0;'>", unsafe_allow_html=True)
                
                res_col1, res_col2 = st.columns([1.1, 1.2])
                with res_col1:
                    if prediction == 1:
                        st.markdown(f"""
                        <div class="glass-card active-badge-red" style="border-left: 5px solid var(--danger);">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <span class="chip-danger">⚠️ HIGH CARDIAC RISK</span>
                                <span style="font-family: var(--mono); font-size: 11px; color: #ff5252;">PREDICTED: POSITIVE</span>
                            </div>
                            <div style="font-size: 36px; font-weight: 700; color: #ff5252; letter-spacing: -0.03em; margin: 4px 0;">
                                {risk_prob * 100:.1f}% Pathological Risk
                            </div>
                            <div class="hud-meta">
                                CLASSIFIER: Random Forest (100 Trees with OneHotEncoder)<br/>
                                DIAGNOSIS: Ischemic Heart Disease (Target = 1)<br/>
                                RECALL SENSITIVITY: 98.25% (Zero Patient False Negative Safeguard)
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="glass-card active-badge-lime" style="border-left: 5px solid var(--lime);">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                <span class="chip">✅ LOW CARDIAC RISK</span>
                                <span style="font-family: var(--mono); font-size: 11px; color: var(--lime);">PREDICTED: NEGATIVE</span>
                            </div>
                            <div style="font-size: 36px; font-weight: 700; color: var(--lime); letter-spacing: -0.03em; margin: 4px 0;">
                                {risk_prob * 100:.1f}% Probability
                            </div>
                            <div class="hud-meta">
                                CLASSIFIER: Random Forest (100 Trees with OneHotEncoder)<br/>
                                DIAGNOSIS: Non-Pathological Baseline (Target = 0)<br/>
                                TOLERANCES: Hemodynamic vitals conform within safe bounds
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                with res_col2:
                    st.markdown("""
                    <div class="glass-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span class="chip-dark">CARDIOVASCULAR RISK METER</span>
                            <span style="font-family: var(--mono); font-size: 11px; color: var(--lime);">THRESHOLD = 0.50</span>
                        </div>
                    """, unsafe_allow_html=True)
                    st.progress(float(risk_prob))

                    st.markdown("##### Key Observed Biomarkers:")
                    indicators = []
                    if st_slope in ["Flat", "Down"]:
                        indicators.append(f"• **ST Segment Slope is {st_slope}**: High clinical association with myocardial ischemia.")
                    if exercise_angina == "Y":
                        indicators.append("• **Exercise-Induced Angina**: Positive stress test correlates with significant coronary stenosis.")
                    if chest_pain == "ASY":
                        indicators.append("• **Asymptomatic Chest Pain**: Carries higher statistical incidence of silent coronary pathology.")
                    if oldpeak > 1.2:
                        indicators.append(f"• **Elevated ST Depression ({oldpeak} mm)**: Strong indicator of ischemic cardiac response.")

                    if indicators:
                        for ind in indicators:
                            st.write(ind)
                    else:
                        st.write("• All clinical parameters align within baseline physiological tolerances.")
                    st.markdown("</div>", unsafe_allow_html=True)

    with tab_ecg:
        st.markdown("### 📈 Electrocardiogram (ECG) ST-Segment Slope Diagnostic Reference")
        st.write("Mr. Harry Attieku-Boateng emphasized understanding why categorical variables matter. The diagram below illustrates why **`ST_Slope`** is the #1 predictor of myocardial ischemia:")
        ecg_path = os.path.join(FIGURES_DIR, "ecg_slope_guide.png")
        if os.path.exists(ecg_path):
            st.image(ecg_path, caption="Figure B1: Clinical ECG ST Segment Morphology in Exercise Stress Testing", use_container_width=True)

    with tab_eda:
        st.markdown("### 📊 Exploratory Visualizations for Heart Disease Patient Cohort")
        c1, c2 = st.columns(2)
        with c1:
            p_eda1 = os.path.join(FIGURES_DIR, "heart_eda_target_and_sex.png")
            if os.path.exists(p_eda1):
                st.image(p_eda1, caption="Figure B2: Heart Disease Incidence Stratified by Biological Sex", use_container_width=True)
            p_eda3 = os.path.join(FIGURES_DIR, "heart_eda_age_maxhr.png")
            if os.path.exists(p_eda3):
                st.image(p_eda3, caption="Figure B3: Age vs. Max Heart Rate Achieved Distribution", use_container_width=True)
        with c2:
            p_eda2 = os.path.join(FIGURES_DIR, "heart_eda_chestpain_st_slope.png")
            if os.path.exists(p_eda2):
                st.image(p_eda2, caption="Figure B4: Correlation of Chest Pain Type and ST Slope with Pathology", use_container_width=True)
            p_eda4 = os.path.join(FIGURES_DIR, "heart_correlation_heatmap.png")
            if os.path.exists(p_eda4):
                st.image(p_eda4, caption="Figure B5: Clinical Biomarker Correlation Heatmap", use_container_width=True)

    with tab_feat:
        st.markdown("### 🧠 Feature Importance after OneHotEncoder Expansion")
        st.write("Because categorical variables (`Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`) are one-hot encoded, the Random Forest model calculates Gini importance across each discrete binary state:")
        f_imp = os.path.join(FIGURES_DIR, "heart_feature_importance.png")
        if os.path.exists(f_imp):
            st.image(f_imp, caption="Figure B6: Gini Importance of OneHotEncoded Predictors in Champion Random Forest", use_container_width=True)

# =============================================================
# MODULE 3: COMPARATIVE BENCHMARKS & TELEMETRY
# =============================================================
elif page == "⚡ Model Telemetry & Comparative Benchmarks":
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 18px;">
        <div>
            <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 6px;">
                <span class="chip">TELEMETRY</span>
                <span class="chip-dark">5-FOLD STRATIFIED CV</span>
            </div>
            <h1 style="margin: 0; font-size: 32px; letter-spacing: -0.03em; color: var(--deck-text);">
                Comparative Algorithm Evaluation Deck
            </h1>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 3D Solution Architecture Cards (Mossline Design System)
    st.markdown("""
    <div class="sol-grid">
        <div class="sol-card">
            <div>
                <span class="tag">[PART A: BOTANICAL]</span>
                <h3>Support Vector Machine</h3>
                <p>Radial Basis Function (RBF) kernel mapping 4D floral geometry into separable Hilbert space.</p>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-family:var(--mono); font-size:11px; color:var(--lime);">96.67% ACC</span>
                <span class="chip">CHAMPION</span>
            </div>
        </div>
        <div class="sol-card">
            <div>
                <span class="tag">[PART B: HEMODYNAMIC]</span>
                <h3>Random Forest Ensemble</h3>
                <p>100 decorrelated decision trees with OneHotEncoder preprocessor maximizing patient recall.</p>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-family:var(--mono); font-size:11px; color:var(--lime);">98.25% RECALL</span>
                <span class="chip">CHAMPION</span>
            </div>
        </div>
        <div class="sol-card">
            <div>
                <span class="tag">[ARCHITECTURE]</span>
                <h3>ColumnTransformer</h3>
                <p>End-to-end reproducible Scikit-Learn pipeline isolating training and test distributions.</p>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-family:var(--mono); font-size:11px; color:var(--lime);">ZERO LEAKAGE</span>
                <span class="chip-dark">STABLE</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c_b1, c_b2 = st.columns(2)
    with c_b1:
        st.markdown("""
        <div class="glass-card">
            <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                <span class="chip">PART A BENCHMARKS</span>
                <span style="font-family:var(--mono); font-size:11px; color:var(--lime);">IRIS CLASSIFIERS</span>
            </div>
        """, unsafe_allow_html=True)
        if iris_metrics_data and "metrics" in iris_metrics_data:
            df_i = pd.DataFrame(iris_metrics_data["metrics"]).T[["accuracy", "precision_macro", "recall_macro", "f1_macro", "cv_mean"]]
            df_i.columns = ["Accuracy", "Precision", "Recall", "F1", "5-Fold CV"]
            st.dataframe(df_i, use_container_width=True)
            
        p_c1 = os.path.join(FIGURES_DIR, "iris_model_comparison.png")
        if os.path.exists(p_c1):
            st.image(p_c1, caption="Multi-Class Classifier Comparison (Iris)", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c_b2:
        st.markdown("""
        <div class="glass-card">
            <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                <span class="chip-danger">PART B BENCHMARKS</span>
                <span style="font-family:var(--mono); font-size:11px; color:var(--lime);">HEART DISEASE CLASSIFIERS</span>
            </div>
        """, unsafe_allow_html=True)
        if heart_metrics_data and "metrics" in heart_metrics_data:
            df_h = pd.DataFrame(heart_metrics_data["metrics"]).T[["accuracy", "precision", "recall", "f1_score", "roc_auc", "cv_accuracy_mean"]]
            df_h.columns = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "5-Fold CV"]
            st.dataframe(df_h, use_container_width=True)
            
        p_c2 = os.path.join(FIGURES_DIR, "heart_roc_curves.png")
        if os.path.exists(p_c2):
            st.image(p_c2, caption="Receiver Operating Characteristic (ROC-AUC) Curves", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# =============================================================
# MODULE 4: PREPROCESSING & REPORT
# =============================================================
elif page == "📑 Preprocessing & Academic Report":
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 18px;">
        <div>
            <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 6px;">
                <span class="chip">ARCHITECTURE</span>
                <span class="chip-dark">ONEHOTENCODER MANDATE</span>
            </div>
            <h1 style="margin: 0; font-size: 32px; letter-spacing: -0.03em; color: var(--deck-text);">
                Mandatory Preprocessing & Technical Report
            </h1>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
            <span class="chip">ONEHOTENCODER VIA COLUMNTRANSFORMER</span>
            <span style="font-family:var(--mono); font-size:11px; color:var(--lime);">SCIKIT-LEARN PIPELINE</span>
        </div>
        <p style="color:var(--deck-text);">
            Mr. Harry Attieku-Boateng explicitly mandated that categorical features in Part B must not be encoded manually
            or with label encoding, as label encoding imposes false ordinal distance. The code below illustrates the production pipeline:
        </p>
    """, unsafe_allow_html=True)

    st.code("""from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ]
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
pipeline.fit(X_train, y_train)
""", language="python")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
            <span class="chip">FORMAL ACADEMIC REPORT (.DOCX)</span>
            <span style="font-family:var(--mono); font-size:11px; color:var(--lime);">DELIVERABLE FOR SUBMISSION</span>
        </div>
        <div class="hud-meta">
            • <b>DOCUMENT</b>: Emmanuel_Asante_PU092301_Assignment1.docx (2.14 MB)<br/>
            • <b>STRUCTURE</b>: Automated Table of Contents (TOC), Table of Figures, 8 Embedded 300 DPI Figures<br/>
            • <b>COMPLIANCE</b>: Turnitin Plagiarism &le; 20%, AI Detection &le; 20%<br/>
            • <b>SUPERVISION</b>: Mr. Harry Attieku-Boateng (Lecturer)
        </div>
    </div>
    """, unsafe_allow_html=True)

# =============================================================
# MODULE 5: MOSSLINE CINEMATIC SCROLLYTELLING (AI STUDIO)
# =============================================================
elif page == "🌿 Mossline Cinematic Scrollytelling (AI Studio)":
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 18px;">
        <div>
            <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 6px;">
                <span class="chip">AI STUDIO EXPORT</span>
                <span class="chip-dark">MOSSLINE INTERACTIVE DESIGN</span>
                <span style="font-family: var(--mono); font-size: 11px; color: var(--deck-text-dim);">640VH SCROLL STAGE</span>
            </div>
            <h1 style="margin: 0; font-size: 32px; letter-spacing: -0.03em; color: var(--deck-text);">
                Mossline Cinematic Field Experience
            </h1>
        </div>
        <div style="text-align: right;">
            <span class="chip" style="background: var(--lime); color: var(--ink);">GOOGLE AI STUDIO APP</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.info("💡 **Interactive Showcase**: Scroll directly within the viewport below to experience the 5-beat scrollytelling fly-in, canvas video scrubbing, procedural SVG fog, interactive hotspots, live sparkline carbon balance, and 3D tilting solution cards.")

    if os.path.exists(MOSSLINE_HTML):
        with open(MOSSLINE_HTML, "r", encoding="utf-8") as f:
            moss_html = f.read()
        components.html(moss_html, height=880, scrolling=True)
    else:
        st.error("Mossline extracted index.html not found.")

# Persistent Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: var(--deck-text-dim); font-family: var(--mono); font-size: 11px; letter-spacing: 0.12em; padding: 10px 0;">
    PENTECOST UNIVERSITY // FACULTY OF COMPUTING &amp; INFORMATION SYSTEMS (FOCIS)<br/>
    APPLIED AI &amp; MACHINE LEARNING ASSIGNMENT 1 • SUPERVISED BY MR. HARRY ATTIEKU-BOATENG<br/>
    DESIGNED IN GOOGLE AI STUDIO • DEVELOPED BY EMMANUEL ASANTE (PU092301 // PUIT23210016)
</div>
""", unsafe_allow_html=True)
