
# 🧠 Adult Income Classification – JNations 2025

This project trains and compares machine learning models (Logistic Regression and Decision Tree) to predict whether a person earns more than $50K/year using the UCI Adult Income dataset. It features:

- Modularized ML pipeline (data → training → evaluation → logging)
- Data preprocessing (cleaning, encoding, scaling)
- Model training and evaluation
- Confusion matrix visualization for each model
- MLflow logging and model registry integration
- Custom experiment logging (not using the default)
- CI/CD with GitHub Actions
- Hosted MLflow tracking server (via Railway)

---

## 🚀 Quickstart

### 1. Fork & Clone and Setup

First, make sure you clone the repository. Then, run the following commands:

```bash
git clone https://github.com/<YOUR_GITHUB_WORKSPACE>/JNations-2025.git
cd JNations-2025

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set up your MLFLOW_TRACKING_URI
GitHub Actions uses `secrets.MLFLOW_TRACKING_URI`.  
Local `.env` is used with `python-dotenv`.

#### Local tracking: `.env` file
```bash
echo "MLFLOW_TRACKING_URI=https://<YOUR-MLFLOW-SERVER>.up.railway.app" > .env
```

#### Github tracking:
1. Go to repository > Setting
2. Secrets and variables
3. Press on Actions
4. Press on New repository sercret and add `MLFLOW_TRACKING_URI` with your URI

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

**Outputs:**
- `outputs/metrics.txt` – F1 scores for each model
- `outputs/confusion_matrix_logistic_regression.png`
- `outputs/confusion_matrix_decision_tree.png`
- MLflow-registered models with version and stage

---

## 🚀 Running Locally / Deploying

### 1. Run MLflow Locally
You can simply run MLflow locally using the following command:

```bash
mlflow server --host 0.0.0.0 --port 8080
```

If you want local persistance, indicate a simple sqlite. Additionally, you can indicate where artifacts are stored:
```bash
mlflow server --host 0.0.0.0 --port 8080 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

### 2. Run MLflow with Railway
If you fancy hosting your MLflow and make it accessible, then let's host it on Railway (demo purposes).
1. Go to [Railway](www.railway.com]), register an account (using Github is convenient)
2. You will get a free plan with $5, enough for this workshop :)
3. Create a project
4. Press on `+ Create` button top right
5. Choose template > look for `MLflow Tracking`
6. Deploy > you will get a URI (aka `MLFLOW_TRACKING_URI`)
## 🤖 CI/CD Pipeline

On every push to `main`:

- `main.py` is executed
- Metrics and models are logged to MLflow
- Confusion matrices and metrics are uploaded as artifacts

```yaml
env:
  MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
```

Artifacts:
- 📄 `metrics.txt`
- 📊 confusion matrices
- 📦 model `.pkl` files (from MLflow artifacts)
