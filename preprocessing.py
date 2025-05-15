import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(csv_path, scaler=None, fit_scaler=True):
    """
    Preprocess Adult Income CSV data:
    - Loads CSV
    - Maps income to binary
    - Replaces '?' with NaN and drops missing
    - One-hot encodes categorical columns
    - Scales numeric features (with optional external scaler)
    
    Returns:
    X: preprocessed feature matrix
    y: binary target (if present)
    scaler: fitted or passed-in StandardScaler
    """
    # Load and clean
    df = pd.read_csv(csv_path)
    df['income_higher_than_50k'] = df['income'].map({'<=50K': 0, '>50K': 1})
    df = df.replace('?', np.nan).dropna()

    # One-hot encode categoricals
    cat_cols = df.select_dtypes('object').columns.drop('income')
    df_enc = pd.get_dummies(df, columns=cat_cols)

    # Scale numerics
    num_cols = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    if scaler is None:
        scaler = StandardScaler()
    if fit_scaler:
        df_enc[num_cols] = scaler.fit_transform(df_enc[num_cols])
    else:
        df_enc[num_cols] = scaler.transform(df_enc[num_cols])

    # Extract features and target
    X = df_enc.drop(columns=['income', 'income_higher_than_50k'], errors='ignore')
    y = df_enc['income_higher_than_50k'] if 'income_higher_than_50k' in df_enc.columns else None

    return X, y, scaler