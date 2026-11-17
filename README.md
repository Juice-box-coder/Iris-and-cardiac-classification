# 🌸 ❤️ Applied AI & Machine Learning Classification Suite
**Course**: Applied AI / Machine Learning & Software Development  
**Lecturer**: Mr. Harry Attieku-Boateng  
**Student Name**: Emmanuel Asante  
**Student ID**: PU092301  
**Deliverable**: Assignment 1 (10 Marks)  

---

## 📌 Links to Deliverables
* 🌐 **Live Deployed Web Application (Vercel)**: [https://assignment1classification.vercel.app](https://assignment1classification.vercel.app)
* 🌐 **Streamlit Mirror**: [https://harry-ai-classification.streamlit.app](https://harry-ai-classification.streamlit.app)
* 📁 **GitHub Source Code Repository**: [https://github.com/emmanuel-asante/applied-ai-classification](https://github.com/emmanuel-asante/applied-ai-classification)
* 📄 **Formal Academic Report**: `Emmanuel_Asante_PU092301_Assignment1.docx`

---

## 📋 Project Scope & Implementation Breakdown

This repository provides production-ready machine learning pipelines and an interactive web inference application for **Assignment 1**:

### 🌸 Part A: Iris Flower Classification (Multi-Class)
* **Dataset**: Built-in Scikit-Learn Iris dataset (`sklearn.datasets.load_iris`).
* **Classes**: `Iris setosa`, `Iris versicolor`, `Iris virginica`.
* **Exploratory Data Analysis**: Pairwise scatter matrix across morphological dimensions, Pearson correlation heatmap.
* **Evaluated Algorithms**:
  * Support Vector Classifier (RBF Kernel) — **Champion Model (96.67% Test Accuracy, 0.9666 Macro F1)**
  * Logistic Regression (Multinomial L2) — 93.33% Accuracy
  * K-Nearest Neighbors (k=5) — 93.33% Accuracy
  * Random Forest (100 Trees) — 90.00% Accuracy
* **Validation**: Stratified 5-Fold Cross-Validation, Confusion Matrices, and Classification Reports.

### ❤️ Part B: Heart Disease Risk Prediction (Binary Classification)
* **Dataset**: Heart Disease diagnostic dataset (~500 records).
* **Target Variable**: `HeartDisease` (`0 = No Disease`, `1 = Heart Disease`).
* **Mandatory Preprocessing Requirement**:
  * Implemented Scikit-Learn’s `OneHotEncoder` via `ColumnTransformer` to encode all nominal features (`Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`).
  * Continuous features (`Age`, `RestingBP`, `Cholesterol`, `FastingBS`, `MaxHR`, `Oldpeak`) scaled with `StandardScaler` and median imputation.
* **Evaluated Algorithms**:
  * Random Forest Classifier — **Champion Model (96.00% Test Accuracy, 98.25% Recall, 0.9812 ROC-AUC)**
  * Logistic Regression — 96.00% Accuracy, 97.96% ROC-AUC
  * Support Vector Classifier (SVC) — 95.00% Accuracy, 97.51% ROC-AUC
  * Gradient Boosting Classifier — 91.00% Accuracy, 95.84% ROC-AUC
* **Feature Explainability**: Gini importance extraction for one-hot encoded features.

---

## 🚀 Quickstart & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/emmanuel-asante/applied-ai-classification.git
cd applied-ai-classification
```

### 2. Set Up Virtual Environment & Dependencies
```bash
# Using Python 3.11+
python -m venv .venv

# Activate environment
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Retrain Models & Generate Plots
```bash
# Train Part A (Iris)
python train_iris.py

# Train Part B (Heart Disease)
python train_heart.py
```

### 4. Generate the Microsoft Word Report (.docx)
```bash
python generate_report.py
```

### 5. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🌐 1-Click Cloud Deployment (Streamlit Cloud)

1. Push this repository to your GitHub account:
   ```bash
   git init
   git add .
   git commit -m "Initial commit of Assignment 1 classification suite"
   git remote add origin https://github.com/<your-username>/applied-ai-classification.git
   git branch -M main
   git push -u origin main
   ```
2. Go to [https://share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
3. Click **"New App"**, select this repository, set the main file path to `app.py`, and click **Deploy**.
4. Copy your live deployment URL (e.g. `https://<your-app-name>.streamlit.app`) and paste it into `generate_report.py` and the report header.

---

## 👥 Adding the Lecturer as a GitHub Collaborator

As explicitly instructed by Mr. Attieku-Boateng:
1. Navigate to your repository on GitHub.
2. Click **Settings** > **Collaborators** > **Add people**.
3. Enter Mr. Harry Attieku-Boateng’s GitHub username or email and click **Add collaborator**.

---

## 📊 Summary of Evaluation Metrics

### Holdout Test Benchmarks:
| Task | Champion Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | 5-Fold CV |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Part A: Iris** | Support Vector Classifier (RBF) | **96.67%** | 0.9697 | 0.9667 | 0.9666 | N/A | 96.67% (±3.12%) |
| **Part B: Heart** | Random Forest (+ OneHotEncoder) | **96.00%** | 0.9492 | **98.25%** | **0.9655** | **0.9812** | 91.25% (±3.06%) |
