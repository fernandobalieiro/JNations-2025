import pandas as pd
import mlflow.sklearn

model = mlflow.sklearn.load_model("models:/MyWorkshopModel/Production")

new_data = pd.read_csv("data/new_input.csv")
predictions = model.predict(new_data)
print(predictions)
