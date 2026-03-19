import torch
import numpy as np
import pandas as pd

from network_anomaly_detection.evaluate import change_labels_to_binary


def get_reconstruction_errors(model, dataloader, device: torch.device) -> np.ndarray:
    """Compute reconstruction error for each sample."""
    
    model = model.to(device)

    model.eval()
    errors = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)
            outputs = model(inputs)
            batch_errors = torch.mean((outputs - inputs) ** 2, dim=1)
            errors.extend(batch_errors.cpu().numpy())

    return np.array(errors)


def predict_anomalies(errors: np.ndarray, threshold: float) -> np.ndarray:
    """Convert reconstruction errors into binary anomaly predictions."""

    y_pred = (errors > threshold).astype(int)

    return y_pred


def select_threshold(errors: np.ndarray, y_true: pd.Series) -> float:
    """Select optimal threshold based on validation set."""

    y_true_binary = change_labels_to_binary(y_true)

    best_threshold = 0.0
    best_f1 = 0.0

    for threshold in np.linspace(errors.min(), errors.max(), 100):
        y_pred = predict_anomalies(errors, threshold)
        tp = np.sum((y_pred == 1) & (y_true_binary == 1))
        fp = np.sum((y_pred == 1) & (y_true_binary == 0))
        fn = np.sum((y_pred == 0) & (y_true_binary == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold



