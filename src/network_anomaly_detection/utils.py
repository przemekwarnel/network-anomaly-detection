import numpy as np
import pandas as pd


def change_labels_to_binary(y: pd.Series) -> np.ndarray:
    """Convert labels to binary (0 = normal, 1 = anomaly)."""
    
    return (y != "normal").astype(int).to_numpy()