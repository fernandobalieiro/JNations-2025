import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from mlflow.tracking import MlflowClient

# ─── Data Load ──────────────────────────────────────────────
df = pd.read_csv("data/adult.csv")
df['income_higher_than_50k'] = df['income'].map({'<=50K': 0, '>50K': 1})
df = df.replace('?', np.nan)
df = df.dropna()

# ─── One-Hot Encode ─────────────────────────────────────────
cat_cols = df.select_dtypes('object').columns.drop('income')
df_enc = pd.get_dummies(df, columns=cat_cols)

# ─── Scaling ────────────────────────────────────────────────
num_cols = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
scaler = StandardScaler()
df_enc[num_cols] = scaler.fit_transform(df_enc[num_cols])

# ─── Train/Test Split ───────────────────────────────────────
X = df_enc.drop(['income', 'income_higher_than_50k'], axis=1)
y = df_enc['income_higher_than_50k']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ─── Model Training ─────────────────────────────────────────
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)

model_dt = DecisionTreeClassifier(max_depth=5, random_state=42)
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)

# ─── MLflow Logging ─────────────────────────────────────────
mlflow.set_experiment("adult-income-workshop")
client = MlflowClient()

# Log Logistic Regression
with mlflow.start_run(run_name="logistic-regression") as run_lr:
    mlflow.log_param("max_iter", 1000)
    mlflow.log_metric("accuracy", acc_lr)
    mlflow.log_metric("f1_score", f1_lr)
    mlflow.sklearn.log_model(model_lr, "model", registered_model_name="LogisticRegression-AdultIncome-Model")
    run_id_lr = run_lr.info.run_id

# Get latest version of LogisticRegression model
latest_lr = client.get_latest_versions("LogisticRegression-AdultIncome-Model", stages=["None"])[-1]

# Log Decision Tree
with mlflow.start_run(run_name="decision-tree") as run_dt:
    mlflow.log_param("max_depth", 5)
    mlflow.log_metric("accuracy", acc_dt)
    mlflow.log_metric("f1_score", f1_dt)
    mlflow.sklearn.log_model(model_dt, "model", registered_model_name="DecisionTree-AdultIncome-Model")
    run_id_dt = run_dt.info.run_id

# Get latest version of DecisionTree model
latest_dt = client.get_latest_versions("DecisionTree-AdultIncome-Model", stages=["None"])[-1]

# ─── Auto-Promotion Logic ───────────────────────────────────
if f1_lr >= f1_dt:
    print(f"✅ Promoting Logistic Regression (F1: {f1_lr:.4f}) to Production")
    client.transition_model_version_stage(
        name="LogisticRegression-AdultIncome-Model",
        version=latest_lr.version,
        stage="Production",
        archive_existing_versions=True
    )
else:
    print(f"✅ Promoting Decision Tree (F1: {f1_dt:.4f}) to Production")
    client.transition_model_version_stage(
        name="DecisionTree-AdultIncome-Model",
        version=latest_dt.version,
        stage="Production",
        archive_existing_versions=True
    )
