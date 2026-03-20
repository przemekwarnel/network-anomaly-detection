import pandas as pd

from network_anomaly_detection.data_split import split_data


def test_split_data():
    df_input = pd.DataFrame({
        "feature1": list(range(16)),
        "feature2": list(range(16, 32)),
        "outcome": ["normal", "attack"] * 8
    })

    train_df, val_df, test_df = split_data(df_input, random_state=42)

    # Train contains only normal traffic
    assert set(train_df["outcome"].unique()) == {"normal"}

    # Validation set contains both normal and anomalies
    assert not val_df.empty
    assert "normal" in val_df["outcome"].values
    assert "attack" in val_df["outcome"].values

    # Test set contains both normal and anomalies
    assert not test_df.empty
    assert "normal" in test_df["outcome"].values
    assert "attack" in test_df["outcome"].values