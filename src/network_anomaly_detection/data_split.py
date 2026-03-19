import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split


def split_data(
        df: pd.DataFrame,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset for anomaly detection.

    Train set contains only normal traffic.
    Validation set contains normal traffic and anomalies for tuning the threshold.
    Test set contains normal traffic and anomalies.
    """
    
    if not (0 < train_size < 1 and 0 < val_size < 1 and 0 < test_size < 1):
        raise ValueError("All split sizes must be between 0 and 1.")

    if not (train_size + val_size + test_size) == 1.0:
        raise ValueError("train_size + val_size + test_size must sum to 1.")
    
    # Separate normal traffic from anomalies 
    df_normal = df[df["outcome"] == "normal"]
    df_anomaly = df[df["outcome"] != "normal"]

    # Split normal traffic into training, validation, and test sets
    train_normal, temp_normal = train_test_split(
        df_normal,
        test_size=val_size + test_size,
        random_state=random_state
    )
    val_normal, test_normal = train_test_split(
        temp_normal,
        test_size=test_size / (val_size + test_size),
        random_state=random_state
    )

    # Split anomalies into validation and test sets
    val_anomaly, test_anomaly = train_test_split(
        df_anomaly,
        test_size=test_size / (val_size + test_size),
        random_state=random_state
    )

    # Validation set = normal + anomalies
    val_df = pd.concat([val_normal, val_anomaly], ignore_index=True)
    val_df = val_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Test set = normal + anomalies
    test_df = pd.concat([test_normal, test_anomaly], ignore_index=True)
    test_df = test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return train_normal, val_df, test_df