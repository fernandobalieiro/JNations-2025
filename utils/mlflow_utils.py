import os
import mlflow
import mlflow.sklearn
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

def log_and_register_models(
    models, X_train, y_train, X_test, y_test, experiment_name="AdultIncomeExperiment"
):
    client = mlflow.tracking.MlflowClient()
    mlflow.sklearn.autolog()

    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    # Create or get the experiment ID
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    with open("outputs/f1_score.txt", "w") as score_file:
        for name, model in models.items():
            with mlflow.start_run(run_name=name, experiment_id=experiment_id):
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                score = f1_score(y_test, preds)
                
                accuracy = accuracy_score(y_test, preds)

                mlflow.log_metric("f1_score", score)
                mlflow.log_metric("accuracy_score", accuracy)
                mlflow.sklearn.log_model(model, "model")

                model_name = f"{name}-AdultIncome-Model"
                result = mlflow.register_model(
                    f"runs:/{mlflow.active_run().info.run_id}/model", model_name
                )

                print(f"✅ Registered {model_name} (F1: {score:.4f}) as version {result.version}")
                score_file.write(f"{name}: {score:.4f}\n")
