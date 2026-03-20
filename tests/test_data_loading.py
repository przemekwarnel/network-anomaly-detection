import pandas as pd
import pytest

from network_anomaly_detection.data_loading import load_data


def test_load_data(tmp_path):
    # Create a small dummy dataset
    df_input = pd.DataFrame({
        "feature1": [1, 2],
        "feature2": [3, 4],
        "outcome": ["normal", "attack"]
    })

    file_path = tmp_path / "test.csv"
    df_input.to_csv(file_path, index=False)

    # Load data
    df = load_data(file_path)

    # Assertions
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "outcome" in df.columns


def test_load_data_raises_without_outcome(tmp_path):
    df_input = pd.DataFrame({
        "feature1": [1, 2]
    })

    file_path = tmp_path / "test.csv"
    df_input.to_csv(file_path, index=False)

    with pytest.raises(ValueError):
        load_data(file_path)