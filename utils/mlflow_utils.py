import os
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, classification_report
from fairlearn.metrics import MetricFrame, true_positive_rate, false_positive_rate

def evaluate_equalized_odds(y_true, y_pred, sensitive_features):
    """
    Evaluate equalized odds fairness metrics (TPR and FPR by group)
    """
    # Create MetricFrame for TPR and FPR
    metric_frame = MetricFrame(
        metrics={
            'TPR': true_positive_rate,
            'FPR': false_positive_rate
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )
    
    # Calculate gaps and ratios
    tpr_by_group = metric_frame.by_group['TPR']
    fpr_by_group = metric_frame.by_group['FPR']
    
    # Calculate gaps (difference between max and min)
    tpr_gap = tpr_by_group.max() - tpr_by_group.min()
    fpr_gap = fpr_by_group.max() - fpr_by_group.min()
    
    # Calculate ratios (min/max)
    tpr_ratio = tpr_by_group.min() / tpr_by_group.max() if tpr_by_group.max() > 0 else 0
    fpr_ratio = fpr_by_group.min() / fpr_by_group.max() if fpr_by_group.max() > 0 else 0
    
    # Create summary
    summary = pd.DataFrame({
        'TPR_gap': [tpr_gap],
        'TPR_ratio': [tpr_ratio],
        'FPR_gap': [fpr_gap],
        'FPR_ratio': [fpr_ratio]
    })
    
    return metric_frame, summary


def create_fairness_plots(metric_frame, gender_summary, model_name):
    """
    Create fairness visualization plots
    """
    # Create the gender comparison bar plot
    gender_long = metric_frame.by_group.reset_index().melt(
        id_vars='gender',
        value_vars=['TPR','FPR'],
        var_name='Metric',
        value_name='Rate'
    )

    plt.figure(figsize=(10, 4))
    
    # Plot 1: TPR and FPR by Gender
    plt.subplot(1, 2, 1)
    sns.barplot(
        data=gender_long,
        x='Metric',
        y='Rate',
        hue='gender',
        errorbar=None
    )
    plt.ylim(0, 1)
    plt.title(f"{model_name}: TPR & FPR by Gender")
    plt.ylabel("Rate")
    plt.legend(title='Gender')
    
    # Plot 2: Summary table as a heatmap
    plt.subplot(1, 2, 2)
    summary_table = pd.DataFrame({
        'TPR': [gender_summary.at[0, 'TPR_gap'], gender_summary.at[0, 'TPR_ratio']],
        'FPR': [gender_summary.at[0, 'FPR_gap'], gender_summary.at[0, 'FPR_ratio']]
    }, index=['Gap', 'Ratio'])
    
    sns.heatmap(summary_table, annot=True, fmt='.3f', cmap='RdYlBu_r', center=0.5)
    plt.title(f"{model_name}: Fairness Summary")
    plt.ylabel("Metric Type")
    
    plt.tight_layout()
    
    # Save plot
    plot_path = f"outputs/{model_name}_fairness_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def log_and_register_models(
    models, X_train, y_train, X_test, y_test, gender_test, experiment_name="AdultIncomeExperiment"
):
    client = mlflow.tracking.MlflowClient()
    mlflow.sklearn.autolog()

    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    # Create or get the experiment ID
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    with open("outputs/f1_score.txt", "w") as score_file:
        for name, model in models.items():
            with mlflow.start_run(run_name=name, experiment_id=experiment_id):
                # Models are already trained, just predict
                preds = model.predict(X_test)
                
                # Calculate basic metrics
                f1 = f1_score(y_test, preds)
                accuracy = accuracy_score(y_test, preds)
                
                # Calculate fairness metrics
                metric_frame, gender_summary = evaluate_equalized_odds(y_test, preds, gender_test)
                
                # Extract individual fairness metrics
                tpr_gap = gender_summary.at[0, 'TPR_gap']
                tpr_ratio = gender_summary.at[0, 'TPR_ratio']
                fpr_gap = gender_summary.at[0, 'FPR_gap']
                fpr_ratio = gender_summary.at[0, 'FPR_ratio']
                
                # Log basic metrics
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("accuracy", accuracy)
                
                # Log fairness metrics
                mlflow.log_metric("tpr_gap", tpr_gap)
                mlflow.log_metric("tpr_ratio", tpr_ratio)
                mlflow.log_metric("fpr_gap", fpr_gap)
                mlflow.log_metric("fpr_ratio", fpr_ratio)
                
                # Log TPR and FPR by gender group
                for gender in metric_frame.by_group.index:
                    mlflow.log_metric(f"tpr_{gender}", metric_frame.by_group.loc[gender, 'TPR'])
                    mlflow.log_metric(f"fpr_{gender}", metric_frame.by_group.loc[gender, 'FPR'])
                
                # Create and log fairness plots
                plot_path = create_fairness_plots(metric_frame, gender_summary, name)
                mlflow.log_artifact(plot_path)
                
                # Log fairness summary as CSV
                summary_path = f"outputs/{name}_fairness_summary.csv"
                gender_summary.to_csv(summary_path, index=False)
                mlflow.log_artifact(summary_path)
                
                # Log detailed metrics by group
                by_group_path = f"outputs/{name}_metrics_by_group.csv"
                metric_frame.by_group.to_csv(by_group_path)
                mlflow.log_artifact(by_group_path)
                
                # Log model
                mlflow.sklearn.log_model(model, "model")

                # Print results (same format as train_models.py)
                print(f"{name} model accuracy:", accuracy)
                print(f"{name} fairness metrics:")
                print(f"  TPR Gap: {tpr_gap:.4f}, TPR Ratio: {tpr_ratio:.4f}")
                print(f"  FPR Gap: {fpr_gap:.4f}, FPR Ratio: {fpr_ratio:.4f}")
                print(classification_report(y_test, preds))

                model_name = f"{name}-AdultIncome-Model"
                result = mlflow.register_model(
                    f"runs:/{mlflow.active_run().info.run_id}/model", model_name
                )

                print(f"✅ Registered {model_name} (F1: {f1:.4f}, Accuracy: {accuracy:.4f}) as version {result.version}")
                score_file.write(f"{name}: F1={f1:.4f}, Accuracy={accuracy:.4f}, TPR_Gap={tpr_gap:.4f}, FPR_Gap={fpr_gap:.4f}\n")
