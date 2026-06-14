from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


METRIC_NAMES = ["accuracy", "precision", "recall", "f1"]


def evaluate_model(model, X_test, y_test) -> Dict[str, float]:
    """Evaluate a model on test data and return core metrics."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label="REAL", zero_division=0),
        "recall": recall_score(y_test, y_pred, pos_label="REAL", zero_division=0),
        "f1": f1_score(y_test, y_pred, pos_label="REAL", zero_division=0),
        "classification_report": classification_report(y_test, y_pred, zero_division=0, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=["FAKE", "REAL"]),
    }


def compare_models(model_metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Convert a dictionary of metrics into a comparison table."""
    rows = []
    for model_name, metrics in model_metrics.items():
        rows.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )
    return pd.DataFrame(rows).sort_values(by="f1", ascending=False).reset_index(drop=True)


def plot_confusion_matrix(cm, labels, output_path: str = None):
    """Plot and optionally save a confusion matrix heatmap."""
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200)
        plt.close()
    return plt
