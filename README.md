# 🧠 Adult Income Classification (ML Workshop)

This project trains and compares machine learning models to predict whether a person's income exceeds \$50K/year using demographic data. It includes:

- Exploratory Data Analysis (EDA)
- Preprocessing (encoding, scaling)
- Two ML models: Logistic Regression & Decision Tree
- Auto-selection and promotion of best model using MLflow
- CI/CD pipeline with GitHub Actions
- Artifact logging: metrics, confusion matrix, model binaries

---

## 🚀 Quickstart

### 1. Clone Install
```bash
git clone git@github.com:JiDarwish/JNations-2025.git
cd JNations-2025

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### 2. Train and Compare Models
```bash
python train.py
```

- Trains Logistic Regression and Decision Tree models
- Logs both to MLflow
- Compares F1 scores and promotes the best one to **Production**
- Saves `.pkl` files and a confusion matrix
- Writes out `metrics.txt` with summary stats

---

### 3. Launch MLflow UI (Optional)
```bash
mlflow ui
```
Visit: [http://localhost:5000](http://localhost:5000)

You’ll see:
- Logged parameters and metrics
- Run comparison
- Model versions and registry

---

### [TODO] 4. Serve Best Model via REST API
```bash
mlflow models serve -m "models:/DecisionTree-AdultIncome-Model/Production" -p 5001
```

Make a prediction:
```bash
curl -X POST http://localhost:5001/invocations   -H "Content-Type: application/json"   -d '{
        "dataframe_split": {
          "columns": [...],
          "data": [[...]]
        }
      }'
```

---

### 5. CI/CD via GitHub Actions

On every `push` to `main`, GitHub Actions will:

- Install dependencies
- Run `train.py`
- Compare models and promote the best one
- Upload the following artifacts:
  - `metrics.txt`
  - `confusion_matrix.png`
  - `*.pkl` model files

You can download them directly from the workflow run UI.

---

## 📦 Requirements

- Python 3.9+
- pandas, scikit-learn, mlflow, matplotlib, joblib

Install with:
```bash
pip install -r requirements.txt
```