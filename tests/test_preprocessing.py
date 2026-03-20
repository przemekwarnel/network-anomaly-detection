import pandas as pd
from sklearn.preprocessing import StandardScaler

from network_anomaly_detection.preprocessing import preprocess_data


def test_preprocess_data():
    df_train = pd.DataFrame({
        "feature1": [1, 2, 3, 4],
        "feature2": [10, 11, 12, 13],
        "feature3": ["A", "B", "A", "B"],
        "constant_feature": [1, 1, 1, 1],
        "outcome": ["normal", "normal", "normal", "normal"],
    })

    df_val = pd.DataFrame({
        "feature1": [5, 6],
        "feature2": [14, 15],
        "feature3": ["A", "C"],
        "constant_feature": [1, 1],
        "outcome": ["normal", "attack"],
    })

    df_test = pd.DataFrame({
        "feature1": [7, 8],
        "feature2": [16, 17],
        "feature3": ["B", "C"],
        "constant_feature": [1, 1],
        "outcome": ["normal", "attack"],
    })

    X_train_scaled, X_val_scaled, X_test_scaled, y_val, y_test, scaler = preprocess_data(
        df_train,
        df_val,
        df_test,
        target_column="outcome"
    )

    assert list(X_train_scaled.columns) == list(X_val_scaled.columns) == list(X_test_scaled.columns)

    assert "outcome" not in X_train_scaled.columns
    assert "outcome" not in X_val_scaled.columns
    assert "outcome" not in X_test_scaled.columns

    assert "constant_feature" not in X_train_scaled.columns
    assert "constant_feature" not in X_val_scaled.columns
    assert "constant_feature" not in X_test_scaled.columns

    assert len(y_val) == len(X_val_scaled)
    assert len(y_test) == len(X_test_scaled)

    assert isinstance(scaler, StandardScaler)

    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in X_train_scaled.dtypes)