from data.load_data import load_and_preprocess
from models.train_model import train_models
from models.evaluate_model import evaluate_and_save
from utils.mlflow_utils import log_and_register_best_model

X_train, X_test, y_train, y_test = load_and_preprocess()
models = train_models(X_train, y_train)
best_model = evaluate_and_save(models, X_test, y_test)
log_and_register_best_model(best_model, X_test, y_test)
