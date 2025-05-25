import os
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt


def save_confusion_matrix(cm, model_name):
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.title(f"Confusion Matrix – {model_name}")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"outputs/confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
    plt.close()


def evaluate_and_save(models, X_test, y_test):
    os.makedirs("outputs", exist_ok=True)

    best_f1 = 0
    best_model = None

    with open("outputs/metrics.txt", "w") as f:
        for name, model in models.items():
            preds = model.predict(X_test)
            score = f1_score(y_test, preds)
            f.write(f"{name}: F1 = {score:.4f}\n")

            cm = confusion_matrix(y_test, preds)
            save_confusion_matrix(cm, name)

            if score > best_f1:
                best_f1 = score
                best_model = model

    return best_model
