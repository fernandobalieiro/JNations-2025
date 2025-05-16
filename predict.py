# import pandas as pd
# import mlflow.sklearn
# from preprocessing import preprocess_data

# model = mlflow.sklearn.load_model("models:/LogisticRegression-AdultIncome-Model/Production")

# new_data = pd.read_csv("data/new_sample.csv")

# X_test, _, _ = preprocess_data("data/new_sample.csv", scaler=scaler, fit_scaler=False)

# predictions = model.predict(new_data)
# print(predictions)