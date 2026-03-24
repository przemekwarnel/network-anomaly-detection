import numpy as np
import pandas as pd

from network_anomaly_detection.detect import predict_anomalies, select_threshold


def test_predict_anomalies():
    errors = np.array([0.1, 0.5, 0.3, 0.8, 0.2])
    threshold = 0.4

    expected_predictions = np.array([0, 1, 0, 1, 0])
    predictions = predict_anomalies(errors, threshold) 

    assert np.array_equal(predictions, expected_predictions)


def test_select_threshold():
    errors = np.array([0.1, 0.5, 0.3, 0.8, 0.2])
    y_true = pd.Series(["normal", "attack", "normal", "attack", "normal"])

    threshold = select_threshold(errors, y_true) 

    assert errors.min() <= threshold <= errors.max()