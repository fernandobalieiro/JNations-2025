# main.py

from data.load_data import load_and_preprocess
from models.build_model import build_models
from utils.mlflow_utils import log_and_register_models
from dotenv import load_dotenv


def main():
    load_dotenv()

    print("🚀 Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess()

    print("🧠 Building models...")
    models = build_models()

    print("📦 Running MLflow: Logging and registering the best model...")
    log_and_register_models(models, X_train, y_train, X_test, y_test)

    print("✅ Pipeline complete.")


if __name__ == "__main__":
    main()
