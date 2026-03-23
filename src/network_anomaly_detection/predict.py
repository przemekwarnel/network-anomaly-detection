import argparse
import pickle
from pathlib import Path

import pandas as pd
import torch
import yaml

from network_anomaly_detection.model import Autoencoder
from network_anomaly_detection.preprocessing import preprocess_new_data
from network_anomaly_detection.detect import get_reconstruction_errors, predict_anomalies
from network_anomaly_detection.train import create_dataloader, get_device


def main():
    """Run inference pipeline for network anomaly detection."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument(
        "--model_dir",
        default="models",
        help="Directory where model, scaler, feature columns, and threshold are saved"
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    # Load artifacts
    with open(model_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    with open(model_dir / "features.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    
    with open(model_dir / "threshold.yaml", "r") as f:
        threshold = yaml.safe_load(f)["threshold"]
    
    # Load data
    df = pd.read_csv(args.input)

    # Preprocess data
    X_new = preprocess_new_data(df, scaler, feature_columns, target_column="outcome")

    # Load model
    input_dim = X_new.shape[1]
    model = Autoencoder(input_dim, hidden_dims=[32, 16], latent_dim=8)  # Use same architecture as training
    model.load_state_dict(torch.load(model_dir / "autoencoder.pth"))

    device = get_device()
    data_loader = create_dataloader(X_new, batch_size=256, shuffle=False)

    # Inference
    errors = get_reconstruction_errors(model, data_loader, device)
    predictions = predict_anomalies(errors, threshold)

    # Output
    results = df.copy()
    results["anomaly_score"] = errors
    results["prediction"] = predictions

    print(results.head())


if __name__ == "__main__":
    main()