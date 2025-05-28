#!/usr/bin/env python3
"""
train_and_mitigate.py

This script reproduces, end-to-end, the preprocessing, baseline model training,
fairness mitigation (ExponentiatedGradient with EqualizedOdds on gender), and
serialization exactly as in the workshop notebook.
"""

import os
import pickle

# ─── 1) Environment setup & imports ──────────────────────────────────────────
# (Make sure you've done: pip install numpy pandas scikit-learn seaborn fairlearn)
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt  # only if you want to visualize
import seaborn as sns            # ditto

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import accuracy_score, classification_report

from fairlearn.reductions    import ExponentiatedGradient, EqualizedOdds

# ─── 2) Load & clean the data ────────────────────────────────────────────────
URL = (
    "https://raw.githubusercontent.com/"
    "saravrajavelu/Adult-Income-Analysis/master/adult.csv"
)
RANDOM_STATE = 42
df = pd.read_csv(URL)

# Map income to binary target
df['income_higher_than_50k'] = df['income'].map({'<=50K':0, '>50K':1})

# Replace '?' placeholders with NaN and drop missing
df.replace(' ?', np.nan, inplace=True)
df.dropna(inplace=True)

# ─── 3) One-Hot Encode Categoricals ─────────────────────────────────────────
# drop the original 'income' string column and sensitive 'gender' from features
cat_cols = df.select_dtypes('object').columns.drop(['income', 'gender'])
df_enc = pd.get_dummies(df, columns=cat_cols)

# ─── 4) Scale Numeric Features ──────────────────────────────────────────────
num_cols = ['age','fnlwgt','educational-num','capital-gain','capital-loss','hours-per-week']
scaler = StandardScaler()
df_enc[num_cols] = scaler.fit_transform(df_enc[num_cols])

# ─── 5) Split into X / y and train/test ──────────────────────────────────────
X = df_enc.drop(columns=['income','income_higher_than_50k','gender'])
y = df_enc['income_higher_than_50k']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# Also extract the sensitive feature series for Fairlearn
gender_train = df.loc[X_train.index, 'gender']
gender_test  = df.loc[X_test.index,  'gender']

# ─── 6) Train baseline Logistic Regression ─────────────────────────────────
base_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
base_model.fit(X_train, y_train)

y_pred_base = base_model.predict(X_test)
print("Baseline model accuracy:", accuracy_score(y_test, y_pred_base))
print(classification_report(y_test, y_pred_base))

# ─── 7) Train fairness-aware mitigator ──────────────────────────────────────
mitigator = ExponentiatedGradient(
    estimator=base_model,
    constraints=EqualizedOdds(),
    eps=0.01,
    # random_state=RANDOM_STATE
)
mitigator.fit(
    X_train,
    y_train,
    sensitive_features=gender_train
)

y_pred_fair = mitigator.predict(X_test)
print("Fair model accuracy:", accuracy_score(y_test, y_pred_fair))
print(classification_report(y_test, y_pred_fair))

# ─── 8) Serialize models ─────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

with open("models/logistic_regression_baseline.pkl", "wb") as f:
    pickle.dump(base_model, f)
print("✅ Saved baseline model → models/logistic_regression_baseline.pkl")

with open("models/fair_mitigator_gender.pkl", "wb") as f:
    pickle.dump(mitigator, f)
print("✅ Saved fairness-aware model → models/fair_mitigator_gender.pkl")


df_enc.to_csv("data/adult_processed.csv", index=False)