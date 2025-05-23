import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_and_preprocess():
    df = pd.read_csv("data/adult.csv")  # adjust path
    df = df.dropna()
    X = df.drop("income", axis=1)
    y = df["income"].apply(lambda x: 1 if x == ">50K" else 0)

    # Example preprocessing pipeline
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])

    X_processed = preprocessor.fit_transform(X)

    return train_test_split(X_processed, y, test_size=0.2, random_state=42)
