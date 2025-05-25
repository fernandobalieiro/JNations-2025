import os
import mlflow.pyfunc

model_path = "models:/LogisticRegression-AdultIncome-Model"
mlflow.pyfunc.serve(
    model_uri=model_path, host="0.0.0.0", port=int(os.environ.get("PORT", 5000))
)
