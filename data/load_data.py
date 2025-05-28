import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess():
    # Load data from the same URL as in train_models.py
    URL = (
        "https://raw.githubusercontent.com/"
        "saravrajavelu/Adult-Income-Analysis/master/adult.csv"
    )
    RANDOM_STATE = 42
    df = pd.read_csv(URL)

    # Map income to binary target (same as train_models.py)
    df['income_higher_than_50k'] = df['income'].map({'<=50K':0, '>50K':1})

    # Replace '?' placeholders with NaN and drop missing
    df.replace(' ?', np.nan, inplace=True)
    df.dropna(inplace=True)

    # One-Hot Encode Categoricals (drop the original 'income' string column and sensitive 'gender' from features)
    cat_cols = df.select_dtypes('object').columns.drop(['income', 'gender'])
    df_enc = pd.get_dummies(df, columns=cat_cols)

    # Scale Numeric Features
    num_cols = ['age','fnlwgt','educational-num','capital-gain','capital-loss','hours-per-week']
    scaler = StandardScaler()
    df_enc[num_cols] = scaler.fit_transform(df_enc[num_cols])

    # Split into X / y and train/test
    X = df_enc.drop(columns=['income','income_higher_than_50k','gender'])
    y = df_enc['income_higher_than_50k']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    # Also extract the sensitive feature series for Fairlearn
    gender_train = df.loc[X_train.index, 'gender']
    gender_test = df.loc[X_test.index, 'gender']

    return X_train, X_test, y_train, y_test, gender_train, gender_test
