"""
Assignment 1 - Part B: Heart Disease Prediction Binary Classification Pipeline
Course: Applied AI / Machine Learning & Software Development
Lecturer: Mr. Harry Attieku-Boateng

Mandatory Lecturer Requirement:
- Categorical features MUST be encoded using scikit-learn's OneHotEncoder.
- Target Variable: HeartDisease (0 = No, 1 = Yes).
- Dataset: ~494 to 500 rows.

This script implements:
1. Loading the Heart Disease dataset (~500 records).
2. Comprehensive Exploratory Data Analysis (EDA) and visualizations.
3. Preprocessing using ColumnTransformer with OneHotEncoder for categoricals
   and StandardScaler for numerical features.
4. Stratified Train/Test Split (80/20).
5. Model Training & Comparison (Logistic Regression, Random Forest, SVC, Gradient Boosting).
6. Comprehensive Evaluation: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix.
7. Feature Importance extraction for one-hot encoded features.
8. Model persistence and evaluation metrics export for web application & report generation.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "heart.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

for d in [FIGURES_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

def run_heart_pipeline():
    print("=" * 60)
    print("PART B: HEART DISEASE PREDICTION PIPELINE")
    print("=" * 60)

    # 1. Load Dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Columns:", df.columns.tolist())

    # Target & Features
    target_col = "HeartDisease"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' missing from dataset.")

    categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
    numerical_cols = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]

    # Verify columns exist
    missing_cat = [c for c in categorical_cols if c not in df.columns]
    missing_num = [c for c in numerical_cols if c not in df.columns]
    if missing_cat or missing_num:
        raise ValueError(f"Missing columns: cat={missing_cat}, num={missing_num}")

    print(f"\nCategorical features ({len(categorical_cols)}): {categorical_cols}")
    print(f"Numerical features ({len(numerical_cols)}): {numerical_cols}")
    print(f"Target distribution:\n{df[target_col].value_counts(normalize=True).rename({0: 'No Disease (0)', 1: 'Heart Disease (1)'}) * 100}\n")

    # 2. Exploratory Data Analysis (EDA)
    sns.set_theme(style="whitegrid", palette="deep")

    # Figure B1: Target Distribution & Gender vs Heart Disease
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.countplot(data=df, x=target_col, ax=axes[0], palette=["#2ecc71", "#e74c3c"])
    axes[0].set_title("Heart Disease Class Distribution (0 = No, 1 = Yes)", fontsize=11, weight="bold")
    axes[0].set_xlabel("Heart Disease Diagnosis", fontsize=10)
    axes[0].set_ylabel("Patient Count", fontsize=10)
    axes[0].set_xticklabels(["No Disease (0)", "Heart Disease (1)"])

    sns.countplot(data=df, x="Sex", hue=target_col, ax=axes[1], palette=["#2ecc71", "#e74c3c"])
    axes[1].set_title("Heart Disease Incidence by Biological Sex", fontsize=11, weight="bold")
    axes[1].set_xlabel("Sex (M = Male, F = Female)", fontsize=10)
    axes[1].set_ylabel("Patient Count", fontsize=10)
    axes[1].legend(["No Disease", "Heart Disease"], loc="upper right")
    plt.tight_layout()
    b1_path = os.path.join(FIGURES_DIR, "heart_eda_target_and_sex.png")
    plt.savefig(b1_path, dpi=300)
    plt.close()
    print(f"Saved: {b1_path}")

    # Figure B2: Chest Pain Type and ST Slope vs Heart Disease
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.countplot(data=df, x="ChestPainType", hue=target_col, ax=axes[0], palette=["#3498db", "#e67e22"])
    axes[0].set_title("Heart Disease by Chest Pain Type (ASY, ATA, NAP, TA)", fontsize=11, weight="bold")
    axes[0].set_xlabel("Chest Pain Type", fontsize=10)
    axes[0].set_ylabel("Patient Count", fontsize=10)
    axes[0].legend(["No Disease", "Heart Disease"])

    sns.countplot(data=df, x="ST_Slope", hue=target_col, ax=axes[1], palette=["#3498db", "#e67e22"])
    axes[1].set_title("Heart Disease by Peak Exercise ST Slope (Up, Flat, Down)", fontsize=11, weight="bold")
    axes[1].set_xlabel("ST Slope", fontsize=10)
    axes[1].set_ylabel("Patient Count", fontsize=10)
    axes[1].legend(["No Disease", "Heart Disease"])
    plt.tight_layout()
    b2_path = os.path.join(FIGURES_DIR, "heart_eda_chestpain_st_slope.png")
    plt.savefig(b2_path, dpi=300)
    plt.close()
    print(f"Saved: {b2_path}")

    # Figure B3: Age & Max Heart Rate Distributions by Diagnosis
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(data=df, x="Age", hue=target_col, kde=True, ax=axes[0], palette=["#2ecc71", "#e74c3c"], element="step")
    axes[0].set_title("Age Distribution by Heart Disease Status", fontsize=11, weight="bold")
    axes[0].set_xlabel("Age (Years)", fontsize=10)
    axes[0].set_ylabel("Density / Count", fontsize=10)

    sns.histplot(data=df, x="MaxHR", hue=target_col, kde=True, ax=axes[1], palette=["#2ecc71", "#e74c3c"], element="step")
    axes[1].set_title("Maximum Heart Rate (MaxHR) by Diagnosis", fontsize=11, weight="bold")
    axes[1].set_xlabel("Max Heart Rate (bpm)", fontsize=10)
    axes[1].set_ylabel("Density / Count", fontsize=10)
    plt.tight_layout()
    b3_path = os.path.join(FIGURES_DIR, "heart_eda_age_maxhr.png")
    plt.savefig(b3_path, dpi=300)
    plt.close()
    print(f"Saved: {b3_path}")

    # Figure B4: Numerical Correlation Matrix
    plt.figure(figsize=(7, 5.5))
    num_corr = df[numerical_cols + [target_col]].corr()
    sns.heatmap(num_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=1, square=True)
    plt.title("Correlation Matrix of Numerical Clinical Indicators", fontsize=12, weight="bold", pad=12)
    plt.tight_layout()
    b4_path = os.path.join(FIGURES_DIR, "heart_correlation_heatmap.png")
    plt.savefig(b4_path, dpi=300)
    plt.close()
    print(f"Saved: {b4_path}")

    # 3. Stratified Train-Test Split (80/20)
    X = df[numerical_cols + categorical_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training split: {X_train.shape[0]} samples | Test split: {X_test.shape[0]} samples")

    # 4. Mandatory Preprocessing Pipeline using OneHotEncoder
    # As explicitly required by the lecturer:
    # "compute for categorical variables... There is something called OneHotEncoder... use it."
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols)
        ],
        remainder="drop"
    )

    # 5. Model Definitions
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42),
        "Support Vector Classifier (SVC)": SVC(kernel="rbf", probability=True, C=1.0, random_state=42),
        "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    }

    metrics_summary = {}
    fitted_pipelines = {}
    roc_data = {}

    for name, clf in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])

        # Fit model
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)

        # 5-fold cross-validation on training data
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": round(float(roc_auc), 4)}

        metrics_summary[name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(roc_auc), 4),
            "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
            "cv_accuracy_std": round(float(cv_scores.std()), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, target_names=["No Disease", "Heart Disease"], output_dict=True)
        }
        fitted_pipelines[name] = pipeline

        print(f"\n--- {name} ---")
        print(f"Accuracy:  {acc * 100:.2f}% | Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f} | F1-Score:  {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
        print(f"CV 5-Fold: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")

    # 6. Comparative Bar Plot
    comp_df = pd.DataFrame([
        {
            "Model": m,
            "Accuracy": metrics_summary[m]["accuracy"],
            "Precision": metrics_summary[m]["precision"],
            "Recall": metrics_summary[m]["recall"],
            "F1-Score": metrics_summary[m]["f1_score"],
            "ROC-AUC": metrics_summary[m]["roc_auc"]
        }
        for m in metrics_summary
    ])

    plt.figure(figsize=(10, 5))
    comp_df.set_index("Model").plot(kind="bar", figsize=(10, 5), colormap="Set2", width=0.75)
    plt.title("Comparative Performance Metrics of Classification Models on Heart Disease Dataset", fontsize=11, weight="bold", pad=12)
    plt.ylabel("Score (0 to 1.0)", fontsize=10)
    plt.ylim(0.70, 1.02)
    plt.xticks(rotation=15, ha="right", fontsize=9)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    b5_path = os.path.join(FIGURES_DIR, "heart_model_comparison.png")
    plt.savefig(b5_path, dpi=300)
    plt.close()
    print(f"Saved: {b5_path}")

    # 7. ROC Curves Plot
    plt.figure(figsize=(7.5, 6))
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]
    for (name, r), c in zip(roc_data.items(), colors):
        plt.plot(r["fpr"], r["tpr"], label=f"{name} (AUC = {r['auc']:.3f})", lw=2, color=c)
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=10)
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=12, weight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    b6_path = os.path.join(FIGURES_DIR, "heart_roc_curves.png")
    plt.savefig(b6_path, dpi=300)
    plt.close()
    print(f"Saved: {b6_path}")

    # 8. Best Model Confusion Matrix
    best_name = max(metrics_summary, key=lambda m: (metrics_summary[m]["f1_score"], metrics_summary[m]["roc_auc"]))
    best_cm = np.array(metrics_summary[best_name]["confusion_matrix"])

    plt.figure(figsize=(6, 5))
    sns.heatmap(best_cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=["No Disease (0)", "Heart Disease (1)"],
                yticklabels=["No Disease (0)", "Heart Disease (1)"],
                cbar=False, annot_kws={"size": 14, "weight": "bold"})
    plt.title(f"Confusion Matrix: {best_name}", fontsize=11, weight="bold", pad=12)
    plt.xlabel("Predicted Condition", fontsize=10, labelpad=8)
    plt.ylabel("Actual Condition", fontsize=10, labelpad=8)
    plt.tight_layout()
    b7_path = os.path.join(FIGURES_DIR, "heart_confusion_matrix.png")
    plt.savefig(b7_path, dpi=300)
    plt.close()
    print(f"Saved: {b7_path}")

    # 9. Feature Importance (from Tree-based Model)
    rf_pipeline = fitted_pipelines["Random Forest Classifier"]
    encoder = rf_pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = encoder.get_feature_names_out(categorical_cols).tolist()
    all_feature_names = numerical_cols + cat_feature_names
    importances = rf_pipeline.named_steps["classifier"].feature_importances_

    feat_df = pd.DataFrame({"Feature": all_feature_names, "Importance": importances})
    feat_df = feat_df.sort_values(by="Importance", ascending=True).tail(12)

    plt.figure(figsize=(8, 5.5))
    plt.barh(feat_df["Feature"], feat_df["Importance"], color="#e74c3c", alpha=0.85, edgecolor="black")
    plt.title("Top 12 Most Predictive Features (Random Forest + OneHotEncoder)", fontsize=11, weight="bold", pad=12)
    plt.xlabel("Relative Gini Importance", fontsize=10)
    plt.tight_layout()
    b8_path = os.path.join(FIGURES_DIR, "heart_feature_importance.png")
    plt.savefig(b8_path, dpi=300)
    plt.close()
    print(f"Saved: {b8_path}")

    # 10. Save Best Model Pipeline
    best_pipeline = fitted_pipelines[best_name]
    model_save_path = os.path.join(MODELS_DIR, "heart_best_pipeline.joblib")
    joblib.dump({
        "pipeline": best_pipeline,
        "best_model_name": best_name,
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "all_feature_names": all_feature_names
    }, model_save_path)
    print(f"Saved best pipeline: {model_save_path}")

    # 11. Save Metrics JSON
    metrics_path = os.path.join(RESULTS_DIR, "heart_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "metrics": metrics_summary,
            "best_model": best_name,
            "numerical_cols": numerical_cols,
            "categorical_cols": categorical_cols,
            "dataset_rows": len(df),
            "roc_data": roc_data
        }, f, indent=4)
    print(f"Saved metrics summary: {metrics_path}")

    print("\nHeart Disease prediction pipeline completed successfully.")
    return metrics_summary

if __name__ == "__main__":
    run_heart_pipeline()
