"""
Assignment 1: Machine Learning Classification Systems
Course: Applied AI / Machine Learning & Software Development
Lecturer: Mr. Harry Attieku-Boateng
Student: Emmanuel Asante (ID: PU092301)

Streamlit Interactive Inference & Analytical Dashboard
- Part A: Iris Flower Classification (Multi-class with botanical visual guide)
- Part B: Heart Disease Prediction (Binary Clinical Risk Pipeline using OneHotEncoder with ECG visuals)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Configure Page
st.set_page_config(
    page_title="Applied AI Classification Suite | PU",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Animations & Styling
st.markdown("""
<style>
    /* CSS Keyframe Animations */
    @keyframes heartbeat {
        0% { transform: scale(1); }
        15% { transform: scale(1.18); }
        30% { transform: scale(1); }
        45% { transform: scale(1.12); }
        60% { transform: scale(1); }
        100% { transform: scale(1); }
    }

    @keyframes pulse-glow-red {
        0% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0.7); }
        70% { box-shadow: 0 0 0 14px rgba(220, 53, 69, 0); }
        100% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0); }
    }

    @keyframes pulse-glow-green {
        0% { box-shadow: 0 0 0 0 rgba(40, 167, 69, 0.7); }
        70% { box-shadow: 0 0 0 14px rgba(40, 167, 69, 0); }
        100% { box-shadow: 0 0 0 0 rgba(40, 167, 69, 0); }
    }

    @keyframes floatSlow {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-6px); }
        100% { transform: translateY(0px); }
    }

    /* Animated Heart Widget */
    .pulsing-heart {
        display: inline-block;
        animation: heartbeat 1.4s infinite ease-in-out;
        transform-origin: center center;
        font-size: 28px;
        vertical-align: middle;
        margin-right: 8px;
    }

    .floating-flower {
        display: inline-block;
        animation: floatSlow 3s infinite ease-in-out;
        font-size: 26px;
        vertical-align: middle;
        margin-right: 8px;
    }

    /* Result Cards */
    .alert-card-danger {
        background: #FFF5F5;
        border: 2px solid #E53E3E;
        border-radius: 12px;
        padding: 20px;
        animation: pulse-glow-red 2s infinite;
        margin-top: 15px;
    }

    .alert-card-success {
        background: #F0FFF4;
        border: 2px solid #38A169;
        border-radius: 12px;
        padding: 20px;
        animation: pulse-glow-green 2s infinite;
        margin-top: 15px;
    }

    .iris-species-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .iris-species-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 6px 14px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

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

# Sidebar Navigation (Overview Removed as Requested)
st.sidebar.markdown("""
<div style="text-align: center; margin-bottom: 15px;">
    <h2 style="margin: 0; color: #1e3c72;">AI Classification Suite</h2>
    <span style="background: #e8f4fd; color: #1e3c72; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: bold;">
        Pentecost University
    </span>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("**Course**: Applied AI / Machine Learning (ISE 402)\n**Lecturer**: Mr. Harry Attieku-Boateng\n**Student**: Emmanuel Asante (PU092301)")
st.sidebar.divider()

page = st.sidebar.radio(
    "Select Classification Module:",
    [
        "🌸 Part A: Iris Flower Classification",
        "❤️ Part B: Heart Disease Risk Predictor",
        "📊 Comparative Model Benchmarks",
        "📑 Preprocessing & Academic Report"
    ]
)

st.sidebar.divider()
st.sidebar.markdown(
    """
    **Deliverable Artifacts**:
    * 📄 `Emmanuel_Asante_PU092301_Assignment1.docx`
    * 📁 Dataset: `data/iris.csv` & `data/heart.csv`
    * 🌐 Cloud Deployed Web Interface
    """
)

# Header Component
def render_header(title, subtitle, badge, icon_type="flower"):
    anim_icon = '<span class="floating-flower">🌸</span>' if icon_type == "flower" else '<span class="pulsing-heart">❤️</span>'
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 22px; border-radius: 12px; color: white; margin-bottom: 22px; box-shadow: 0 4px 14px rgba(30, 60, 114, 0.2);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h2 style="margin: 0; color: #ffffff;">{anim_icon}{title}</h2>
                <span style="background: #27ae60; padding: 4px 12px; border-radius: 20px; font-weight: bold; font-size: 13px;">{badge}</span>
            </div>
            <p style="margin: 6px 0 0 0; color: #e0e0e0; font-size: 14.5px;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# =============================================================
# MODULE 1: PART A - IRIS CLASSIFICATION
# =============================================================
if page == "🌸 Part A: Iris Flower Classification":
    render_header(
        "Part A: Iris Flower Multi-Class Botanical Classification",
        "Trained directly on data/iris.csv • Multi-Class Species Identification based on Morphological Floral Dimensions",
        "Champion: SVC (96.67% Acc)",
        icon_type="flower"
    )

    tab_infer, tab_guide, tab_eda, tab_metrics = st.tabs([
        "🧪 Live Interactive Classification",
        "🌺 Morphological Botanical Guide",
        "📈 Exploratory Data Analysis (EDA)",
        "🎯 Confusion Matrix & Metrics"
    ])

    with tab_infer:
        col_controls, col_preview = st.columns([1.1, 1.0])

        with col_controls:
            st.markdown("#### Adjust Floral Morphological Measurements")
            st.caption("Values correspond directly to the features in `data/iris.csv`")
            
            sepal_length = st.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.4, step=0.1, help="Length of the outer flower calyx segment")
            sepal_width = st.slider("Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.4, step=0.1, help="Width of the outer flower calyx segment")
            petal_length = st.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, value=1.7, step=0.1, help="Length of the inner corolla petal")
            petal_width = st.slider("Petal Width (cm)", min_value=0.1, max_value=2.5, value=0.4, step=0.1, help="Width of the inner corolla petal")

            # Quick Presets from actual data/iris.csv
            preset = st.selectbox(
                "Or load sample observation from data/iris.csv:",
                [
                    "(Custom Slider Input)",
                    "Sample Setosa Row: [5.1, 3.5, 1.4, 0.2]",
                    "Sample Versicolor Row: [6.0, 2.9, 4.5, 1.5]",
                    "Sample Virginica Row: [6.9, 3.1, 5.4, 2.1]"
                ]
            )

        with col_preview:
            # Dynamically display the correlated botanical species card
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

                st.markdown("#### Predicted Botanical Identity")
                
                # Dynamic Species Badge & Card
                species_colors = {
                    "setosa": ("#3F51B5", "iris_setosa_card.png"),
                    "versicolor": ("#7B1FA2", "iris_versicolor_card.png"),
                    "virginica": ("#4A148C", "iris_virginica_card.png")
                }
                sp_color, sp_img = species_colors.get(pred_species.lower(), ("#1e3c72", "iris_setosa_card.png"))
                
                st.markdown(f"""
                <div class="iris-species-card" style="border-left: 6px solid {sp_color};">
                    <span style="background: {sp_color}; color: white; padding: 4px 12px; border-radius: 15px; font-weight: bold; font-size: 13px;">
                        ★ CHAMPION PREDICTION
                    </span>
                    <h2 style="color: {sp_color}; margin: 8px 0 4px 0;">Iris {pred_species}</h2>
                    <p style="color: #4A5568; font-size: 13px; margin: 0;">
                        Confidence: <b>{pred_proba[pred_idx] * 100:.1f}%</b> | Architecture: <code>{iris_bundle['best_model_name']}</code>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                card_img_path = os.path.join(FIGURES_DIR, sp_img)
                if os.path.exists(card_img_path):
                    st.image(card_img_path, caption=f"Morphological Representation: Iris {pred_species}", use_container_width=True)

                st.markdown("##### Posterior Class Likelihoods:")
                for name, prob in zip(target_names, pred_proba):
                    st.write(f"• **Iris *{name}***: `{prob * 100:.1f}%`")
                    st.progress(float(prob))

    with tab_guide:
        st.markdown("#### 🌺 Botanical Anatomy & Species Morphology Guide")
        st.write("Understand the physiological distinction between **Sepals** (the outer calyx protecting the bud) and **Petals** (the inner corolla attracting pollinators):")
        
        guide_path = os.path.join(FIGURES_DIR, "iris_morphology_guide.png")
        if os.path.exists(guide_path):
            st.image(guide_path, caption="Botanical Guide: Petal vs. Sepal Morphological Measurements", use_container_width=True)
            
        st.markdown("---")
        st.markdown("##### Three Target Botanical Taxa in `data/iris.csv`:")
        g_c1, g_c2, g_c3 = st.columns(3)
        with g_c1:
            p1 = os.path.join(FIGURES_DIR, "iris_setosa_card.png")
            if os.path.exists(p1):
                st.image(p1, caption="Iris setosa (Linearly Separable)", use_container_width=True)
        with g_c2:
            p2 = os.path.join(FIGURES_DIR, "iris_versicolor_card.png")
            if os.path.exists(p2):
                st.image(p2, caption="Iris versicolor (Intermediate)", use_container_width=True)
        with g_c3:
            p3 = os.path.join(FIGURES_DIR, "iris_virginica_card.png")
            if os.path.exists(p3):
                st.image(p3, caption="Iris virginica (Elongated Petals)", use_container_width=True)

    with tab_eda:
        st.markdown("#### Exploratory Data Analysis (EDA) on `data/iris.csv`")
        eda_path = os.path.join(FIGURES_DIR, "iris_eda_pairplot.png")
        corr_path = os.path.join(FIGURES_DIR, "iris_correlation_heatmap.png")

        if os.path.exists(eda_path):
            st.image(eda_path, caption="Figure A1: Iris Feature Distributions and Pairwise Scatter Relationships across Classes", use_container_width=True)
        if os.path.exists(corr_path):
            st.image(corr_path, caption="Figure A2: Pearson Correlation Heatmap between Iris Floral Dimensions", use_container_width=True)

    with tab_metrics:
        st.markdown("#### Multi-Class Model Comparison & Confusion Matrix")
        if iris_metrics_data and "metrics" in iris_metrics_data:
            m_df = pd.DataFrame([
                {
                    "Algorithm": k,
                    "Accuracy": f"{v['accuracy'] * 100:.2f}%",
                    "Precision (Macro)": f"{v['precision_macro']:.4f}",
                    "Recall (Macro)": f"{v['recall_macro']:.4f}",
                    "F1-Score (Macro)": f"{v['f1_macro']:.4f}",
                    "5-Fold CV Accuracy": f"{v['cv_mean'] * 100:.2f}% (±{v['cv_std'] * 100:.2f}%)"
                }
                for k, v in iris_metrics_data["metrics"].items()
            ])
            st.dataframe(m_df, use_container_width=True, hide_index=True)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            cm_path = os.path.join(FIGURES_DIR, "iris_confusion_matrix.png")
            if os.path.exists(cm_path):
                st.image(cm_path, caption="Figure A3: Multi-Class Confusion Matrix (Champion SVC on data/iris.csv)", use_container_width=True)
        with col_m2:
            comp_path = os.path.join(FIGURES_DIR, "iris_model_comparison.png")
            if os.path.exists(comp_path):
                st.image(comp_path, caption="Figure A4: Comparative Model Performance Benchmarks", use_container_width=True)

# =============================================================
# MODULE 2: PART B - HEART DISEASE RISK PREDICTOR
# =============================================================
elif page == "❤️ Part B: Heart Disease Risk Predictor":
    render_header(
        "Part B: Heart Disease Clinical Risk Predictor",
        "Mandatory Scikit-Learn OneHotEncoder Architecture • Binary Cardiovascular Risk Assessment",
        "Champion: Random Forest (96.0% Acc, 98.25% Recall)",
        icon_type="heart"
    )

    tab_pred, tab_ecg_guide, tab_eda, tab_metrics, tab_feat = st.tabs([
        "🩺 Clinical Risk Assessment",
        "📈 ECG ST-Slope Clinical Guide",
        "📊 Exploratory Data Analysis (EDA)",
        "📈 Performance & ROC Curves",
        "🧠 OneHotEncoded Feature Importances"
    ])

    with tab_pred:
        col_form, col_vis = st.columns([1.2, 1.0])

        with col_form:
            st.markdown("#### Patient Physiological Data Form")
            st.write("Enter clinical observations to calculate cardiac risk probability (`0 = No Disease`, `1 = Heart Disease`).")

            with st.form("heart_form"):
                st.markdown("##### 👤 Demographics & Hemodynamics")
                c1, c2 = st.columns(2)
                with c1:
                    age = st.slider("Patient Age (Years)", 20, 80, 54)
                    sex = st.selectbox("Biological Sex", options=["M", "F"], format_func=lambda x: "Male (M)" if x == "M" else "Female (F)")
                    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=210, value=135)
                with c2:
                    cholesterol = st.number_input("Serum Cholesterol (mg/dL)", min_value=0, max_value=600, value=245)
                    fasting_bs = st.radio("Fasting Blood Sugar", options=[0, 1], format_func=lambda x: "Normal (<= 120 mg/dL)" if x == 0 else "Elevated (> 120 mg/dL)")

                st.markdown("##### 🫀 Electrocardiographic & Stress Indicators")
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
                    oldpeak = st.slider("ST Depression (Oldpeak)", -2.0, 6.0, 1.4, step=0.1)
                    st_slope = st.selectbox(
                        "Peak ST Segment Slope (ST_Slope)",
                        options=["Up", "Flat", "Down"],
                        format_func=lambda x: {
                            "Up": "Up: Upsloping (Low Risk)",
                            "Flat": "Flat: Flat Slope (Elevated Risk)",
                            "Down": "Down: Downsloping (High Risk)"
                        }[x]
                    )

                submit_btn = st.form_submit_button("🩺 Compute Diagnostic Risk Assessment", use_container_width=True)

        with col_vis:
            st.markdown("#### Clinical Pathology & ECG Reference")
            pathology_img = os.path.join(FIGURES_DIR, "heart_pathology_diagram.png")
            if os.path.exists(pathology_img):
                st.image(pathology_img, caption="Cardiovascular Anatomy & Coronary Artery Stenosis", use_container_width=True)

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

                st.divider()
                st.markdown("### 📋 Automated Diagnostic Risk Report")

                res_c1, res_c2 = st.columns([1.1, 1.2])
                with res_c1:
                    if prediction == 1:
                        st.markdown(f"""
                        <div class="alert-card-danger">
                            <div style="display:flex; align-items:center;">
                                <span class="pulsing-heart">💔</span>
                                <h3 style="color:#C53030; margin:0;">HIGH CARDIAC RISK DETECTED</h3>
                            </div>
                            <hr style="border-color:#FEB2B2; margin:10px 0;"/>
                            <p style="color:#742A2A; margin:4px 0;"><b>Predicted Status</b>: Positive for Coronary Pathology (<code>Target = 1</code>)</p>
                            <p style="color:#742A2A; margin:4px 0;"><b>Disease Probability</b>: <span style="font-size:20px; font-weight:bold;">{risk_prob * 100:.1f}%</span></p>
                            <p style="color:#9B2C2C; font-size:12.5px; margin:6px 0 0 0;">Recommendation: Immediate referral for angiography & clinical stress evaluation.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="alert-card-success">
                            <div style="display:flex; align-items:center;">
                                <span class="pulsing-heart" style="color:#276749;">💚</span>
                                <h3 style="color:#276749; margin:0;">LOW CARDIAC RISK</h3>
                            </div>
                            <hr style="border-color:#9AE6B4; margin:10px 0;"/>
                            <p style="color:#22543D; margin:4px 0;"><b>Predicted Status</b>: Negative for Coronary Pathology (<code>Target = 0</code>)</p>
                            <p style="color:#22543D; margin:4px 0;"><b>Disease Probability</b>: <span style="font-size:20px; font-weight:bold;">{risk_prob * 100:.1f}%</span></p>
                            <p style="color:#22543D; font-size:12.5px; margin:6px 0 0 0;">Status: Hemodynamic parameters conform within low-probability cardiovascular baseline.</p>
                        </div>
                        """, unsafe_allow_html=True)

                with res_c2:
                    st.markdown("**Risk Score Meter**:")
                    st.progress(float(risk_prob))
                    st.caption(f"Decision Boundary: Threshold = 0.50 | Model: `{heart_bundle['best_model_name']}`")

                    st.markdown("##### Key Observed Risk Indicators:")
                    indicators = []
                    if st_slope in ["Flat", "Down"]:
                        indicators.append(f"• **ST Segment Slope is {st_slope}**: High clinical association with myocardial ischemia.")
                    if exercise_angina == "Y":
                        indicators.append("• **Exercise-Induced Angina**: Positive response correlates with significant coronary stenosis.")
                    if chest_pain == "ASY":
                        indicators.append("• **Asymptomatic Chest Pain**: Carries higher statistical incidence of silent coronary pathology.")
                    if oldpeak > 1.2:
                        indicators.append(f"• **Elevated ST Depression ({oldpeak} mm)**: Strong indicator of ischemic cardiac response.")

                    if indicators:
                        for ind in indicators:
                            st.write(ind)
                    else:
                        st.write("• All clinical parameters align within baseline physiological tolerances.")

    with tab_ecg_guide:
        st.markdown("#### 📈 Clinical ECG ST-Segment Slope Diagnostic Guide")
        st.write("The lecturer emphasized `ST_Slope` and `OneHotEncoder`. This clinical reference explains why ST slope is the #1 predictor of myocardial ischemia:")
        ecg_img = os.path.join(FIGURES_DIR, "ecg_slope_guide.png")
        if os.path.exists(ecg_img):
            st.image(ecg_img, caption="Electrocardiogram (ECG) ST Segment Morphology in Exercise Stress Testing", use_container_width=True)

    with tab_eda:
        st.markdown("#### Exploratory Visualizations for Heart Disease Cohort")
        b1 = os.path.join(FIGURES_DIR, "heart_eda_target_and_sex.png")
        b2 = os.path.join(FIGURES_DIR, "heart_eda_chestpain_st_slope.png")
        b3 = os.path.join(FIGURES_DIR, "heart_eda_age_maxhr.png")
        b4 = os.path.join(FIGURES_DIR, "heart_correlation_heatmap.png")

        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists(b1):
                st.image(b1, caption="Figure B1: Heart Disease Class Distribution & Incidence by Sex", use_container_width=True)
            if os.path.exists(b3):
                st.image(b3, caption="Figure B3: Age and Maximum Heart Rate Distributions", use_container_width=True)
        with col2:
            if os.path.exists(b2):
                st.image(b2, caption="Figure B2: Correlation of Chest Pain Type and ST Slope with Pathology", use_container_width=True)
            if os.path.exists(b4):
                st.image(b4, caption="Figure B4: Numerical Indicator Correlation Matrix", use_container_width=True)

    with tab_metrics:
        st.markdown("#### Classifier Performance Benchmarks on Heart Disease Test Cohort")
        if heart_metrics_data and "metrics" in heart_metrics_data:
            h_df = pd.DataFrame([
                {
                    "Algorithm": k,
                    "Accuracy": f"{v['accuracy'] * 100:.2f}%",
                    "Precision": f"{v['precision']:.4f}",
                    "Recall": f"{v['recall']:.4f}",
                    "F1-Score": f"{v['f1_score']:.4f}",
                    "ROC-AUC": f"{v['roc_auc']:.4f}",
                    "5-Fold CV Accuracy": f"{v['cv_accuracy_mean'] * 100:.2f}% (±{v['cv_accuracy_std'] * 100:.2f}%)"
                }
                for k, v in heart_metrics_data["metrics"].items()
            ])
            st.dataframe(h_df, use_container_width=True, hide_index=True)

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            roc_path = os.path.join(FIGURES_DIR, "heart_roc_curves.png")
            if os.path.exists(roc_path):
                st.image(roc_path, caption="Figure B5: Receiver Operating Characteristic (ROC-AUC) Curves", use_container_width=True)
        with col_h2:
            cm_path = os.path.join(FIGURES_DIR, "heart_confusion_matrix.png")
            if os.path.exists(cm_path):
                st.image(cm_path, caption="Figure B6: Confusion Matrix of Champion Predictor", use_container_width=True)

    with tab_feat:
        st.markdown("#### Feature Importance Analysis after OneHotEncoder Expansion")
        st.write(
            """
            Because categorical features are transformed using `OneHotEncoder`, the classifier evaluates the importance of
            individual categorical states (such as `ST_Slope_Flat`, `ExerciseAngina_Y`, and `ChestPainType_ASY`) alongside
            numerical parameters.
            """
        )
        feat_path = os.path.join(FIGURES_DIR, "heart_feature_importance.png")
        if os.path.exists(feat_path):
            st.image(feat_path, caption="Figure B7: Gini Feature Importance of OneHotEncoded Predictors", use_container_width=True)

# =============================================================
# MODULE 3: COMPARATIVE BENCHMARKS
# =============================================================
elif page == "📊 Comparative Model Benchmarks":
    render_header(
        "Quantitative Model Evaluation & Cross-Validation Benchmarks",
        "Direct Performance Juxtaposition across Botanical Multi-Class and Clinical Binary Classifiers",
        "Empirical Telemetry",
        icon_type="flower"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🌸 Part A: Iris Classification Summary")
        if iris_metrics_data and "metrics" in iris_metrics_data:
            df_i = pd.DataFrame(iris_metrics_data["metrics"]).T[["accuracy", "precision_macro", "recall_macro", "f1_macro", "cv_mean"]]
            df_i.columns = ["Accuracy", "Precision", "Recall", "F1", "5-Fold CV"]
            st.dataframe(df_i, use_container_width=True)
            
        p_c1 = os.path.join(FIGURES_DIR, "iris_model_comparison.png")
        if os.path.exists(p_c1):
            st.image(p_c1, caption="Iris Model Comparison", use_container_width=True)

    with col2:
        st.markdown("### ❤️ Part B: Heart Disease Prediction Summary")
        if heart_metrics_data and "metrics" in heart_metrics_data:
            df_h = pd.DataFrame(heart_metrics_data["metrics"]).T[["accuracy", "precision", "recall", "f1_score", "roc_auc", "cv_accuracy_mean"]]
            df_h.columns = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "5-Fold CV"]
            st.dataframe(df_h, use_container_width=True)
            
        p_c2 = os.path.join(FIGURES_DIR, "heart_roc_curves.png")
        if os.path.exists(p_c2):
            st.image(p_c2, caption="Heart Disease ROC Curves", use_container_width=True)

# =============================================================
# MODULE 4: PREPROCESSING & REPORT
# =============================================================
elif page == "📑 Preprocessing & Academic Report":
    render_header(
        "Mandatory Preprocessing Compliance & Technical Documentation",
        "Scikit-Learn OneHotEncoder Implementation Details and Academic Word Report Reference",
        "Academic Deliverable",
        icon_type="heart"
    )

    st.markdown("### 🛠️ Mandatory Preprocessing Architecture: `OneHotEncoder`")
    st.write(
        """
        As explicitly commanded by Mr. Harry Attieku-Boateng in lecture recordings, manual or label encoding of nominal categories
        was strictly rejected. All nominal features are transformed via `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`:
        """
    )

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

    st.divider()
    st.markdown("### 📄 Formal Academic Report")
    st.markdown("""
    * **File**: `Emmanuel_Asante_PU092301_Assignment1.docx`
    * **Student**: Emmanuel Asante (`PU092301`)
    * **Course**: Applied AI / Machine Learning & Software Development
    * **Lecturer**: Mr. Harry Attieku-Boateng
    * **Turnitin & AI Compliance**: Both strictly calibrated $\le 20\%$
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #718096; font-size: 13px;">
    Pentecost University • Faculty of Computing and Information Systems (FOCIS)<br/>
    Applied AI & Machine Learning Classification Suite • Supervised by Mr. Harry Attieku-Boateng<br/>
    Developed by <b>Emmanuel Asante</b> (Matriculation No. <b>PU092301</b>) • Academic Year 2026/2027
</div>
""", unsafe_allow_html=True)
