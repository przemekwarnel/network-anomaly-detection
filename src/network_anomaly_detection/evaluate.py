import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, recall_score, f1_score

from network_anomaly_detection.detect import predict_anomalies


def change_labels_to_binary(y: pd.Series) -> np.ndarray:
    """Convert labels to binary (0 = normal, 1 = anomaly)."""
    
    return (y != "normal").astype(int).to_numpy()

def evaluate_model(
        errors: np.ndarray,
        y_true: pd.Series,
        threshold: float
    ) -> dict[str, float]:
    """Evaluate anomaly detection performance."""

    y_true_binary = change_labels_to_binary(y_true)
    y_pred = predict_anomalies(errors, threshold)

    precision = float(precision_score(y_true_binary, y_pred))
    recall = float(recall_score(y_true_binary, y_pred))
    f1 = float(f1_score(y_true_binary, y_pred))

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }