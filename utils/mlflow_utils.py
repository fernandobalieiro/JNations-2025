import mlflow
import mlflow.sklearn
from sklearn.metrics import f1_score

def log_and_register_best_model(models, X_test, y_test, experiment_name="AdultIncomeExperiment"):
    best_model = None
    best_name = ""
    best_score = 0
    best_version = None

    client = mlflow.tracking.MlflowClient()

    # Create or get the experiment ID
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    for name, model in models.items():
        with mlflow.start_run(run_name=name, experiment_id=experiment_id):
            preds = model.predict(X_test)
            score = f1_score(y_test, preds)

            mlflow.log_metric("f1_score", score)
            mlflow.sklearn.log_model(model, "model")

            model_name = f"{name}-AdultIncome-Model"
            result = mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/model", model_name
            )

            print(f"✅ Registered {model_name} (F1: {score:.4f}) as version {result.version}")

            if score > best_score:
                best_score = score
                best_model = model
                best_name = model_name
                best_version = result.version

    # Promote the best model to Production
    if best_name:
        print(f"🚀 Promoting {best_name} (F1: {best_score:.4f}) to Production")
        client.transition_model_version_stage(
            name=best_name,
            version=best_version,
            stage="Production",
            archive_existing_versions=True
        )
