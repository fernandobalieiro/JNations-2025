import mlflow
import mlflow.sklearn
from sklearn.metrics import f1_score
from models.evaluate_model import evaluate_and_save
# MLFLOW_TRACKING_URI=https://mlflow-tracking-production-f89b.up.railway.app/


def log_and_register_best_model(
    models, X_train, y_train, X_test, y_test, experiment_name="AdultIncomeExperiment"
):
    best_name = ""
    best_score = 0
    best_version = None

    client = mlflow.tracking.MlflowClient()
    mlflow.sklearn.autolog()

    # Create or get the experiment ID
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    for name, model in models.items():
        with mlflow.start_run(run_name=name, experiment_id=experiment_id):
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            score = f1_score(y_test, preds)

            mlflow.log_metric("f1_score", score)
            mlflow.sklearn.log_model(model, "model")

            model_name = f"{name}-AdultIncome-Model"
            result = mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/model", model_name
            )

            print(
                f"✅ Registered {model_name} (F1: {score:.4f}) as version {result.version}"
            )

            if score > best_score:
                best_score = score
                best_name = model_name
                best_version = result.version
