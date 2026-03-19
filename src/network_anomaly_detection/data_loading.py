import pandas as pd


def load_data(data_path: str) -> pd.DataFrame:
    """Load network traffic dataset."""

    df = pd.read_csv(data_path, compression="infer")

    if df.empty:
        raise ValueError("Loaded dataframe is empty.")
    
    if "outcome" not in df.columns:
        raise ValueError("Expected 'outcome' column not found in the dataset.")

    return df