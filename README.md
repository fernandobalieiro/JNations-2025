# 🧠 Adult Income Classification – JNations 2025

This project trains and compares machine learning models (Logistic Regression and Decision Tree) to predict whether a person earns more than $50K/year using the UCI Adult Income dataset. It features:

- Data preprocessing (cleaning, encoding, scaling)
- Model training and evaluation
- Fairness checks
- CI/CD with GitHub Actions
- Hosted MLflow tracking server (via Railway)
- Auto-promotion of the best model
- Artifact logging: metrics, confusion matrix, models

---

## 🚀 Quickstart

### 1. Clone and Setup

```bash
git clone https://github.com/JiDarwish/JNations-2025.git
cd JNations-2025

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set up your `.env` file (for local tracking)

```bash
echo "MLFLOW_TRACKING_URI=https://your-mlflow-server.up.railway.app" > .env
```

> ✅ Your `.env` is only used locally. In CI, MLflow is configured via GitHub Secrets.

---

## 🏋️‍♀️ Run Training

```bash
python train.py
```

This:
- Loads and preprocesses data
- Trains two models
- Logs metrics and models to MLflow
- Promotes the best model to **Production**
- Generates:
  - `metrics.txt`
  - `confusion_matrix.png`
  - model `.pkl` files

---

## 🤖 CI/CD Pipeline

On every push to `main`:

- `train.py` is executed
- Metrics and models are logged to MLflow (hosted)
- Best model is promoted to `Production`
- Artifacts are uploaded for inspection

```yaml
env:
  MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
```

Artifacts:
- 📄 `metrics.txt`
- 📊 `confusion_matrix.png`
- 📦 `model.pkl`

---

## 📈 MLflow UI

Access the hosted MLflow server (on Railway):

```bash
mlflow ui  # for local, or visit https://your-server.up.railway.app
```

Track:
- Parameters (e.g. `max_iter`, `max_depth`)
- Metrics (accuracy, F1)
- Versioned models
- Production deployment history

---

## 📬 Predictions

Use the production model in code:

```python
import mlflow.sklearn
model = mlflow.sklearn.load_model("models:/LogisticRegression-AdultIncome-Model/Production")
preds = model.predict(X_new)
```

Ensure `X_new` is preprocessed just like the training data.

---

## 🔒 Secrets

GitHub Actions uses `secrets.MLFLOW_TRACKING_URI`.  
Local `.env` is used for fallback via `python-dotenv`.
