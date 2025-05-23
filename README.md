
# 🧠 Adult Income Classification – JNations 2025

This project trains and compares machine learning models (Logistic Regression and Decision Tree) to predict whether a person earns more than $50K/year using the UCI Adult Income dataset. It features:

- Modularized ML pipeline (data → training → evaluation → logging)
- Data preprocessing (cleaning, encoding, scaling)
- Model training and evaluation
- Confusion matrix visualization for each model
- MLflow logging and model registry integration
- Auto-promotion of the best model to **Production**
- Custom experiment logging (not using the default)
- CI/CD with GitHub Actions
- Hosted MLflow tracking server (via Railway)

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

## 🏋️‍♀️ Run the ML Pipeline

```bash
python main.py
```

This will:
- Load and preprocess data
- Train Logistic Regression and Decision Tree models
- Evaluate each model, save F1 scores and confusion matrices
- Create or use a named MLflow experiment (`AdultIncomeExperiment`)
- Log both models to MLflow
- Promote the best-performing model (highest F1) to **Production**

**Outputs:**
- `outputs/metrics.txt` – F1 scores for each model
- `outputs/confusion_matrix_logistic_regression.png`
- `outputs/confusion_matrix_decision_tree.png`
- MLflow-registered models with version and stage

---

## 🤖 CI/CD Pipeline

On every push to `main`:

- `main.py` is executed
- Metrics and models are logged to MLflow
- Best model is promoted to `Production`
- Confusion matrices and metrics are uploaded as artifacts

```yaml
env:
  MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
```

Artifacts:
- 📄 `metrics.txt`
- 📊 confusion matrices
- 📦 model `.pkl` files (from MLflow artifacts)

---

## 📈 MLflow UI

Access the hosted MLflow server:

```bash
mlflow ui  # for local, or visit https://your-mlflow-server.up.railway.app
```

Track:
- Parameters and hyperparameters
- Metrics (F1, accuracy)
- Versioned models
- Production deployment stage
- Confusion matrices and experiment runs

---

## 🔒 Secrets

GitHub Actions uses `secrets.MLFLOW_TRACKING_URI`.  
Local `.env` is used with `python-dotenv`.

---

## 🚀 Deploying to Railway / Running Locally

### 1. Deploy MLflow to Railway

- Connect your repo to [Railway](https://railway.app/)
- Add the `MLFLOW_TRACKING_URI` secret
- Add a `Procfile`:

```bash
web: mlflow models serve -m models:/LogisticRegression-AdultIncome-Model/Production -h 0.0.0.0 -p $PORT
```

### 2. Run MLflow Locally

```bash
mlflow ui
export MLFLOW_TRACKING_URI=http://localhost:5000
```

### 3. Train and Register

```bash
python main.py
```

### 4. Inspect

- Go to MLflow UI to compare runs and see the registry

