# main.py

from data.load_data import load_and_preprocess
from models.build_model import build_models
from models.evaluate_model import evaluate_and_save
from utils.mlflow_utils import log_and_register_best_model
from dotenv import load_dotenv


def main():
    load_dotenv()

    print("🚀 Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess()

    print("🧠 Building models...")
    models = build_models()

    print("📦 Running MLflow: Logging and registering the best model...")
    log_and_register_best_model(models, X_train, y_train, X_test, y_test)

    print("✅ Pipeline complete.")


if __name__ == "__main__":
    main()
