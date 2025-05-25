# main.py

from data.load_data import load_and_preprocess
from models.train_model import train_models
from models.evaluate_model import evaluate_and_save
from utils.mlflow_utils import log_and_register_best_model
from dotenv import load_dotenv


def main():
    load_dotenv()

    print("🚀 Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess()

    print("🧠 Training models...")
    models = train_models(X_train, y_train)

    print("📊 Evaluating models...")
    evaluate_and_save(models, X_test, y_test)

    print("📦 Logging and registering the best model to MLflow...")
    log_and_register_best_model(models, X_test, y_test)

    print("✅ Pipeline complete.")


if __name__ == "__main__":
    main()
