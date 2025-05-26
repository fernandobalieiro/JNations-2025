from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def build_models():
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(),
    }
    return models
