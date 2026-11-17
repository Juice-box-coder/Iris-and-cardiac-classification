"""
Assignment 1 - Part A: Iris Flower Classification Pipeline
Course: Applied AI / Machine Learning & Software Development
Lecturer: Mr. Harry Attieku-Boateng
Student: Emmanuel Asante (ID: PU092301)

This script implements:
1. Loading the exact Iris dataset from 'data/iris.csv'.
2. Comprehensive Exploratory Data Analysis (EDA) and visualizations.
3. Train/Test Stratified Split and Feature Scaling.
4. Model Training & Comparison (Logistic Regression, Random Forest, SVM, KNN).
5. Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.
6. Exporting trained models, evaluation metrics, and figures for report generation.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Output paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

for d in [FIGURES_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

def run_iris_pipeline():
    print("=" * 60)
    print("PART A: IRIS FLOWER CLASSIFICATION USING DATA/IRIS.CSV")
    print("=" * 60)

    # 1. Load Dataset from data/iris.csv
    csv_path = os.path.join(DATA_DIR, "iris.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Iris dataset file not found at: {csv_path}")

    print(f"Loading Iris dataset directly from: {csv_path}")
    df = pd.read_csv(csv_path)

    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    target_col = "species"

    # Verify columns exist
    for col in feature_names + [target_col]:
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' not found in {csv_path}. Available columns: {list(df.columns)}")

    X = df[feature_names]
    
    # Target encoding: setosa -> 0, versicolor -> 1, virginica -> 2
    unique_species = sorted(df[target_col].unique().tolist())
    species_to_id = {name: idx for idx, name in enumerate(unique_species)}
    id_to_species = {idx: name for idx, name in enumerate(unique_species)}
    y = df[target_col].map(species_to_id)
    target_names = np.array(unique_species)

    print(f"Dataset shape: {df.shape}")
    print(f"Target classes: {list(target_names)}")
    print(f"Class distribution:\n{df[target_col].value_counts()}\n")

    # 2. Exploratory Data Analysis (EDA) Plots
    sns.set_theme(style="whitegrid", palette="muted")

    # Figure A1: Pairplot of Features by Species
    df_plot = df.copy()
    pairplot = sns.pairplot(df_plot, hue=target_col, markers=["o", "s", "D"], diag_kind="kde", height=2.2)
    pairplot.fig.subplots_adjust(top=0.93)
    pairplot.fig.suptitle("Iris Flower Feature Distributions and Pairwise Relationships", fontsize=13, weight="bold")
    pairplot_path = os.path.join(FIGURES_DIR, "iris_eda_pairplot.png")
    pairplot.savefig(pairplot_path, dpi=300)
    plt.close()
    print(f"Saved: {pairplot_path}")

    # Figure A2: Correlation Heatmap
    plt.figure(figsize=(7, 5.5))
    corr = X.corr()
    sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=1, square=True)
    plt.title("Correlation Heatmap of Iris Morphological Features", fontsize=12, weight="bold", pad=12)
    plt.tight_layout()
    corr_path = os.path.join(FIGURES_DIR, "iris_correlation_heatmap.png")
    plt.savefig(corr_path, dpi=300)
    plt.close()
    print(f"Saved: {corr_path}")

    # 3. Train-Test Split (Stratified 80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # 4. Models to Evaluate
    models = {
        "Support Vector Machine (SVC)": SVC(kernel="rbf", probability=True, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=300, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5)
    }

    metrics_summary = {}
    fitted_pipelines = {}

    for name, clf in models.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", clf)
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # 5-fold cross validation on training set
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")

        metrics_summary[name] = {
            "accuracy": round(float(acc), 4),
            "precision_macro": round(float(prec_macro), 4),
            "recall_macro": round(float(rec_macro), 4),
            "f1_macro": round(float(f1_macro), 4),
            "f1_weighted": round(float(f1_weighted), 4),
            "cv_mean": round(float(cv_scores.mean()), 4),
            "cv_std": round(float(cv_scores.std()), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        }
        fitted_pipelines[name] = pipe

        print(f"\n--- {name} ---")
        print(f"Accuracy:  {acc * 100:.2f}% | Precision (Macro): {prec_macro:.4f}")
        print(f"Recall:    {rec_macro:.4f} | F1-Score (Macro):  {f1_macro:.4f}")
        print(f"CV 5-Fold: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")

    # 5. Model Comparison Plot
    comparison_df = pd.DataFrame([
        {
            "Model": m,
            "Accuracy": metrics_summary[m]["accuracy"],
            "F1-Score": metrics_summary[m]["f1_macro"],
            "CV Accuracy": metrics_summary[m]["cv_mean"]
        }
        for m in metrics_summary
    ])

    plt.figure(figsize=(9, 5))
    comp_plot = comparison_df.set_index("Model").plot(kind="bar", figsize=(9, 5), colormap="viridis", width=0.7)
    plt.title("Comparative Performance of Multi-Class Classifiers on Iris Test Set", fontsize=12, weight="bold", pad=12)
    plt.ylabel("Score (0 to 1.0)", fontsize=11)
    plt.ylim(0.85, 1.02)
    plt.xticks(rotation=15, ha="right", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(loc="lower right")
    plt.tight_layout()
    comp_path = os.path.join(FIGURES_DIR, "iris_model_comparison.png")
    plt.savefig(comp_path, dpi=300)
    plt.close()
    print(f"Saved: {comp_path}")

    # 6. Best Model Confusion Matrix Visualization
    best_model_name = max(metrics_summary, key=lambda m: (metrics_summary[m]["f1_macro"], metrics_summary[m]["cv_mean"]))
    best_cm = np.array(metrics_summary[best_model_name]["confusion_matrix"])

    plt.figure(figsize=(6, 5))
    sns.heatmap(best_cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names,
                cbar=False, annot_kws={"size": 14, "weight": "bold"})
    plt.title(f"Confusion Matrix: {best_model_name}", fontsize=12, weight="bold", pad=12)
    plt.xlabel("Predicted Species", fontsize=11, labelpad=8)
    plt.ylabel("True Species", fontsize=11, labelpad=8)
    plt.tight_layout()
    cm_path = os.path.join(FIGURES_DIR, "iris_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved: {cm_path}")

    # 7. Save Best Pipeline and Metadata
    best_pipeline = fitted_pipelines[best_model_name]
    model_save_path = os.path.join(MODELS_DIR, "iris_best_pipeline.joblib")
    joblib.dump({
        "pipeline": best_pipeline,
        "feature_names": feature_names,
        "target_names": target_names.tolist(),
        "species_to_id": species_to_id,
        "id_to_species": id_to_species,
        "best_model_name": best_model_name
    }, model_save_path)
    print(f"Saved best model: {model_save_path}")

    # 8. Save Metrics JSON
    metrics_path = os.path.join(RESULTS_DIR, "iris_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "metrics": metrics_summary,
            "best_model": best_model_name,
            "target_names": target_names.tolist(),
            "feature_names": feature_names,
            "data_source": "data/iris.csv"
        }, f, indent=4)
    print(f"Saved metrics summary: {metrics_path}")

    print("\nIris classification pipeline executed successfully on data/iris.csv.")
    return metrics_summary

if __name__ == "__main__":
    run_iris_pipeline()
