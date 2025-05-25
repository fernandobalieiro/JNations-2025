from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def train_models(X_train, y_train):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models
