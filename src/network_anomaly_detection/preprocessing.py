import pandas as pd
from typing import Tuple
from sklearn.preprocessing import StandardScaler


def preprocess_data(
        df_train: pd.DataFrame,
        df_val: pd.DataFrame, 
        df_test: pd.DataFrame,
        target_column: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """Preprocess data for anomaly detection."""
    
    # Check if all dataframes have the same columns
    if set(df_train.columns) != set(df_val.columns) or set(df_train.columns) != set(df_test.columns):
        raise ValueError("Train, validation, and test dataframes must have the same columns.")
    
    # Check if the target column is present in all dataframes
    if target_column not in df_train.columns or target_column not in df_val.columns or target_column not in df_test.columns:
        raise ValueError(f"All dataframes must contain the '{target_column}' column.")
    
    # Remove features with zero variance
    zero_var_cols = df_train.columns[df_train.nunique() <= 1]

    df_train = df_train.drop(columns=zero_var_cols)
    df_val = df_val.drop(columns=zero_var_cols)
    df_test = df_test.drop(columns=zero_var_cols)

    # Remove target from training features
    X_train = df_train.drop(columns=[target_column])

    # Separate target from features in test and validation sets
    X_test = df_test.drop(columns=[target_column])
    y_test = df_test[target_column]

    X_val = df_val.drop(columns=[target_column])
    y_val = df_val[target_column]
    
    # One-hot encode categorical features
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns

    X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
    X_val = pd.get_dummies(X_val, columns=categorical_cols, drop_first=True)

    train_columns = X_train.columns
    X_val = X_val.reindex(columns=train_columns, fill_value=0)
    X_test = X_test.reindex(columns=train_columns, fill_value=0)    

    # Fit scaler on training data
    scaler = StandardScaler()
    scaler.fit(X_train)

    # Transform both training and test data
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_train.columns,
        index=X_val.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_train.columns,
        index=X_test.index
    )   

    return X_train_scaled, X_val_scaled, X_test_scaled, y_val, y_test, scaler
