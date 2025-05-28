from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds


def build_models(X_train, y_train, gender_train):
    RANDOM_STATE = 42
    
    # Train baseline Logistic Regression first
    logistic_regression_baseline = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    logistic_regression_baseline.fit(X_train, y_train)
    
    # Use the trained baseline model as the base estimator for the fairness-aware mitigator
    # This matches the Workshop notebook approach: base_est = model (the already trained model)
    fair_mitigator = ExponentiatedGradient(
        estimator=logistic_regression_baseline,  # Use the already trained model
        constraints=EqualizedOdds(),
        eps=0.01,
    )
    fair_mitigator.fit(X_train, y_train, sensitive_features=gender_train)
    
    return {
        "LogisticRegression": logistic_regression_baseline,
        "FairLogisticRegression": fair_mitigator,
    }
