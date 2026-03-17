import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split


def split_train_test(
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset for anomaly detection.

    Train set contains only normal traffic.
    Test set contains normal traffic and anomalies.
    """
    
    # Separate normal traffic from anomalies 
    df_normal = df[df["outcome"] == "normal"]
    df_anomaly = df[df["outcome"] != "normal"]

    # Split normal traffic into training and testing sets
    train_normal, test_normal = train_test_split(
        df_normal,
        test_size=test_size,
        random_state=random_state
    )

    # Test set = normal + anomalies
    test_df = pd.concat([test_normal, df_anomaly], ignore_index=True)

    return train_normal, test_df