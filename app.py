"""
Assignment 1: Machine Learning Classification Systems
Course: Applied AI / Machine Learning & Software Development
Lecturer: Mr. Harry Attieku-Boateng

Streamlit Interactive Inference & Analytical Dashboard
- Part A: Iris Flower Classification (Multi-class)
- Part B: Heart Disease Prediction (Binary Clinical Risk Pipeline using OneHotEncoder)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Configure Page
st.set_page_config(
    page_title="AI Classification Systems | Assignment 1",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Sidebar Navigation
st.sidebar.title("🩺 Navigation")
st.sidebar.markdown("**Course**: Applied AI / ML\n**Lecturer**: Mr. Harry Attieku-Boateng")
st.sidebar.divider()

page = st.sidebar.radio(
    "Select Application Module:",
    [
        "🏠 Overview & System Architecture",
        "🌸 Part A: Iris Flower Classification",
        "❤️ Part B: Heart Disease Risk Predictor",
        "📊 Comparative Model Benchmarks",
        "📑 Preprocessing & Academic Report"
    ]
)

st.sidebar.divider()
st.sidebar.markdown(
    """
    **Project Deliverables**:
    - [GitHub Source Repository](https://github.com)
    - [Live Cloud Application](https://streamlit.io)
    - *Word Report*: `Emmanuel_Asante_Assignment1.docx`
    """
)

# Header Component
def render_header(title, subtitle, badge):
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 22px; border-radius: 10px; color: white; margin-bottom: 22px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h2 style="margin: 0; color: #ffffff;">{title}</h2>
                <span style="background: #27ae60; padding: 4px 12px; border-radius: 20px; font-weight: bold; font-size: 13px;">{badge}</span>
            </div>
            <p style="margin: 6px 0 0 0; color: #e0e0e0; font-size: 15px;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------------------------------------------
# PAGE 1: OVERVIEW & SYSTEM ARCHITECTURE
# -------------------------------------------------------------
if page == "🏠 Overview & System Architecture":
    render_header(
        "Machine Learning Classification Suite",
        "Dual-Pipeline Implementation: Multi-Class Iris Botanical Classification & OneHotEncoded Binary Cardiac Risk Prediction",
        "Assignment 1 • 10 Marks"
    )

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("### 📌 Executive Overview")
        st.write(
            """
            This web application serves as the production inference interface and analytical demonstrator for
            **Assignment 1** in *Applied AI / Machine Learning & Software Development*.

            The project addresses two distinct machine learning challenges:
            1. **Part A (Botanical Multi-Class Task)**: Supervised classification of Iris species (*Iris setosa*, *Iris versicolor*, *Iris virginica*) using standard morphological parameters.
            2. **Part B (Clinical Binary Task)**: Cardiovascular pathology risk prediction over a patient cohort (~500 records) utilizing a strict **`OneHotEncoder`** preprocessing architecture mandated by the course specification.
            """
        )

        st.info(
            """
            **Mandatory Preprocessing Requirement Compliance**:
            As explicitly instructed by the lecturer, categorical features in Part B (`Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`) are strictly preprocessed using Scikit-Learn's `OneHotEncoder` within an isolated `ColumnTransformer` to eliminate ordinal bias and data leakage.
            """
        )

    with col2:
        st.markdown("### 🎓 Student & Submission Details")
        st.markdown(
            """
            | Parameter | Value |
            | :--- | :--- |
            | **Student Name** | Emmanuel Asante |
            | **Course** | Applied AI / Machine Learning |
            | **Lecturer** | Mr. Harry Attieku-Boateng |
            | **Turnitin Threshold** | Similarity $\le$ 20%, AI $\le$ 20% |
            | **Model Framework** | Scikit-Learn 1.9+, Python 3.11 |
            | **Deployment** | Streamlit Cloud / Render |
            """
        )

    st.divider()
    st.markdown("### 🛠️ Pipeline Architectural Workflow")
    st.markdown(
        """
        ```
        [Raw Input Data] 
               │
               ├──> Part A (Iris): Feature Matrix (4 dims) ──> StandardScaler ──> Best SVC / LR Pipeline ──> Botanical Class
               │
               └──> Part B (Heart): Numerical Features (6) ───> StandardScaler ──┐
                                    Categorical Features (5) ─> OneHotEncoder ──┴─> Best Ensemble Pipeline ──> Binary Risk (0 / 1)
        ```
        """
    )

# -------------------------------------------------------------
# PAGE 2: PART A - IRIS CLASSIFICATION
# -------------------------------------------------------------
elif page == "🌸 Part A: Iris Flower Classification":
    render_header(
        "Part A: Iris Flower Multi-Class Classification",
        "Botanical Species Identification based on Morphological Floral Dimensions",
        "Part A • Multi-Class"
    )

    tab_infer, tab_eda, tab_metrics = st.tabs(["🧪 Interactive Classification", "📈 Exploratory Data Analysis (EDA)", "🎯 Performance Evaluation"])

    with tab_infer:
        st.markdown("#### Input Morphological Floral Measurements")
        col_in1, col_in2 = st.columns(2)

        with col_in1:
            sepal_length = st.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.8, step=0.1, help="Calyx segment length")
            sepal_width = st.slider("Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.0, step=0.1, help="Calyx segment width")

        with col_in2:
            petal_length = st.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, value=4.3, step=0.1, help="Corolla inner segment length")
            petal_width = st.slider("Petal Width (cm)", min_value=0.1, max_value=2.5, value=1.3, step=0.1, help="Corolla inner segment width")

        if iris_bundle:
            pipeline = iris_bundle["pipeline"]
            target_names = iris_bundle["target_names"]
            feature_names = iris_bundle["feature_names"]

            input_df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=feature_names)
            pred_idx = pipeline.predict(input_df)[0]
            pred_proba = pipeline.predict_proba(input_df)[0]
            pred_species = target_names[pred_idx]

            st.divider()
            st.markdown("#### Botanical Identification Result")
            res_col1, res_col2 = st.columns([2, 3])

            with res_col1:
                st.success(f"### Predicted Species:\n**Iris {pred_species}**")
                st.caption(f"Champion Model Architecture: `{iris_bundle['best_model_name']}`")

            with res_col2:
                st.markdown("**Posterior Class Probabilities**:")
                for name, prob in zip(target_names, pred_proba):
                    st.write(f"Iris *{name}*: `{prob * 100:.1f}%`")
                    st.progress(float(prob))
        else:
            st.warning("Model bundle not found. Please run `train_iris.py` to generate model artifacts.")

    with tab_eda:
        st.markdown("#### Exploratory Visualizations for Iris Morphological Distribution")
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
                st.image(cm_path, caption="Figure A3: Multi-Class Confusion Matrix (Best Classifier)", use_container_width=True)
        with col_m2:
            comp_path = os.path.join(FIGURES_DIR, "iris_model_comparison.png")
            if os.path.exists(comp_path):
                st.image(comp_path, caption="Figure A4: Comparative Model Performance Benchmarks", use_container_width=True)

# -------------------------------------------------------------
# PAGE 3: PART B - HEART DISEASE RISK PREDICTOR
# -------------------------------------------------------------
elif page == "❤️ Part B: Heart Disease Risk Predictor":
    render_header(
        "Part B: Heart Disease Clinical Risk Predictor",
        "Cardiovascular Pathological Prediction using a scikit-learn OneHotEncoder Pipeline",
        "Part B • Binary Classification"
    )

    tab_pred, tab_eda, tab_metrics, tab_feat = st.tabs([
        "🩺 Clinical Risk Assessment",
        "📊 Exploratory Data Analysis (EDA)",
        "📈 Performance & ROC Curves",
        "🧠 Feature Importances"
    ])

    with tab_pred:
        st.markdown("#### Patient Clinical Observation Form")
        st.write("Enter patient physiological observations to compute binary risk likelihood (`0 = No Disease`, `1 = Heart Disease`).")

        with st.form("heart_form"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("##### 👤 Patient Demographics & Vitals")
                age = st.slider("Patient Age (Years)", 20, 80, 52)
                sex = st.selectbox("Biological Sex", options=["M", "F"], format_func=lambda x: "Male (M)" if x == "M" else "Female (F)")
                resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=210, value=130)
                cholesterol = st.number_input("Serum Cholesterol (mg/dL)", min_value=0, max_value=600, value=220, help="Note: 0 indicates missing/unrecorded baseline in standard protocol")
                fasting_bs = st.radio("Fasting Blood Sugar", options=[0, 1], format_func=lambda x: "Normal (<= 120 mg/dL)" if x == 0 else "Elevated (> 120 mg/dL)")

            with col2:
                st.markdown("##### 🫀 Cardiac Indicators & Stress Test")
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
                    "Resting Electrocardiogram (RestingECG)",
                    options=["Normal", "ST", "LVH"],
                    format_func=lambda x: {
                        "Normal": "Normal: Normal Waveform",
                        "ST": "ST: ST-T Wave Abnormality",
                        "LVH": "LVH: Left Ventricular Hypertrophy"
                    }[x]
                )
                max_hr = st.slider("Maximum Heart Rate Achieved (MaxHR)", 60, 210, 142)
                exercise_angina = st.selectbox("Exercise-Induced Angina", options=["N", "Y"], format_func=lambda x: "No (N)" if x == "N" else "Yes (Y)")
                oldpeak = st.slider("ST Depression Induced by Exercise (Oldpeak)", -2.0, 6.0, 1.2, step=0.1)
                st_slope = st.selectbox(
                    "Peak Exercise ST Segment Slope (ST_Slope)",
                    options=["Up", "Flat", "Down"],
                    format_func=lambda x: {
                        "Up": "Up: Upsloping (Low Risk Indicator)",
                        "Flat": "Flat: Flat ST Slope (Elevated Risk Indicator)",
                        "Down": "Down: Downsloping (High Risk Indicator)"
                    }[x]
                )

            submit_btn = st.form_submit_button("🩺 Compute Diagnostic Risk Assessment", use_container_width=True)

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
                st.markdown("### 📋 Diagnostic Assessment Report")

                res_c1, res_c2 = st.columns([2, 3])
                with res_c1:
                    if prediction == 1:
                        st.error(
                            f"""
                            ### ⚠️ HIGH CARDIAC RISK DETECTED
                            **Predicted Diagnosis**: Positive for Heart Disease (`Target = 1`)  
                            **Estimated Probability**: `{risk_prob * 100:.1f}%`
                            """
                        )
                    else:
                        st.success(
                            f"""
                            ### ✅ LOW CARDIAC RISK
                            **Predicted Diagnosis**: Negative for Heart Disease (`Target = 0`)  
                            **Estimated Probability of Disease**: `{risk_prob * 100:.1f}%`
                            """
                        )

                with res_c2:
                    st.markdown("**Risk Score Meter**:")
                    st.progress(float(risk_prob))
                    st.caption(f"Decision Boundary: Threshold = 0.50 | Model: `{heart_bundle['best_model_name']}`")

                    st.markdown("##### Key Observed Risk Indicators:")
                    indicators = []
                    if st_slope in ["Flat", "Down"]:
                        indicators.append(f"• **ST Segment Slope is {st_slope}**: High clinical association with myocardial ischemia.")
                    if exercise_angina == "Y":
                        indicators.append("• **Exercise-Induced Angina**: Positive test correlates with coronary stenosis.")
                    if chest_pain == "ASY":
                        indicators.append("• **Asymptomatic Chest Pain**: Paradoxically carries higher statistical incidence of undetected coronary pathology.")
                    if oldpeak > 1.5:
                        indicators.append(f"• **Significant ST Depression ({oldpeak})**: Strong predictor of ischemic stress response.")

                    if indicators:
                        for ind in indicators:
                            st.write(ind)
                    else:
                        st.write("• Clinical parameters align within low-probability cardiovascular baseline.")
            else:
                st.warning("Model bundle not found. Please run `train_heart.py` first.")

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

# -------------------------------------------------------------
# PAGE 4: COMPARATIVE MODEL BENCHMARKS
# -------------------------------------------------------------
elif page == "📊 Comparative Model Benchmarks":
    render_header(
        "Cross-Task Performance Benchmarks",
        "Comprehensive Statistical Comparison between Part A (Iris) and Part B (Heart Disease)",
        "Model Synthesis"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌸 Part A: Iris Multi-Class Summary")
        if iris_metrics_data and "metrics" in iris_metrics_data:
            for k, v in iris_metrics_data["metrics"].items():
                with st.expander(f"📌 {k} (Accuracy: {v['accuracy'] * 100:.1f}%)"):
                    st.write(f"- **Precision (Macro)**: {v['precision_macro']:.4f}")
                    st.write(f"- **Recall (Macro)**: {v['recall_macro']:.4f}")
                    st.write(f"- **F1-Score (Macro)**: {v['f1_macro']:.4f}")
                    st.write(f"- **5-Fold Cross Validation**: {v['cv_mean'] * 100:.2f}% (±{v['cv_std'] * 100:.2f}%)")

    with col2:
        st.markdown("### ❤️ Part B: Heart Disease Binary Summary")
        if heart_metrics_data and "metrics" in heart_metrics_data:
            for k, v in heart_metrics_data["metrics"].items():
                with st.expander(f"📌 {k} (Accuracy: {v['accuracy'] * 100:.1f}%)"):
                    st.write(f"- **Precision**: {v['precision']:.4f}")
                    st.write(f"- **Recall**: {v['recall']:.4f}")
                    st.write(f"- **F1-Score**: {v['f1_score']:.4f}")
                    st.write(f"- **ROC-AUC**: {v['roc_auc']:.4f}")
                    st.write(f"- **5-Fold Cross Validation**: {v['cv_accuracy_mean'] * 100:.2f}% (±{v['cv_accuracy_std'] * 100:.2f}%)")

# -------------------------------------------------------------
# PAGE 5: PREPROCESSING & ACADEMIC REPORT
# -------------------------------------------------------------
elif page == "📑 Preprocessing & Academic Report":
    render_header(
        "Methodological Rigor & Submission Deliverables",
        "Verification of Scikit-Learn OneHotEncoder Protocol and Word Documentation",
        "Compliance"
    )

    st.markdown("### 🎯 Mandatory Preprocessing Implementation")
    st.markdown(
        """
        The lecturer strictly required:
        > *"You have to compute for categorical variables. Go and learn it. There is something called OneHotEncoder. Go and read about it and use it. I will not teach you."*

        In `train_heart.py`, this is implemented with production-grade separation:
        """
    )

    st.code(
        '''
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']),
        ('cat', categorical_transformer, ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])
    ]
)
        ''',
        language="python"
    )

    st.success(
        """
        **Why OneHotEncoder is Essential Over Integer/Label Encoding**:
        Categorical nominal features (such as `ChestPainType`: ASY, ATA, NAP, TA) possess no intrinsic natural ordering.
        If mapped arbitrarily to integers 0, 1, 2, 3, linear classifiers and distance-based metrics erroneously infer that `TA` is three times the magnitude of `ASY`.
        `OneHotEncoder` expands each discrete category into independent orthogonal binary vectors, ensuring zero false mathematical bias.
        """
    )
