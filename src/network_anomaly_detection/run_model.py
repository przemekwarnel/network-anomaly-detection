import torch
import argparse
import yaml
from pathlib import Path
import pickle

from network_anomaly_detection.config import load_config
from network_anomaly_detection.data_loading import load_data
from network_anomaly_detection.data_split import split_data
from network_anomaly_detection.preprocessing import preprocess_data
from network_anomaly_detection.model import Autoencoder
from network_anomaly_detection.train import train_model, get_device, create_dataloader
from network_anomaly_detection.detect import get_reconstruction_errors, select_threshold
from network_anomaly_detection.evaluate import evaluate_model
from network_anomaly_detection.visualization import plot_training_loss


def main(): 
    """Run end-to-end training and evaluation pipeline for network anomaly detection."""

    # Load config and params 
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    config = load_config(args.config) 

    data_path = config["data"]["path"]
    target_column = config["data"]["target_column"]
    train_size = config["data"]["train_size"]
    val_size = config["data"]["val_size"]
    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]

    hidden_dims = config["model"]["hidden_dims"]
    latent_dim = config["model"]["latent_dim"]

    batch_size = config["training"]["batch_size"]
    num_epochs = config["training"]["num_epochs"]
    learning_rate = config["training"]["learning_rate"]
    criterion_name = config["training"]["criterion"]
    optimizer_name = config["training"]["optimizer"]

    # Load data
    df = load_data(data_path)

    # Split data into train, validation, and test sets
    train_normal, val_df, test_df = split_data(df, train_size, val_size, test_size, random_state)

    # Preprocess data
    X_train, X_val, X_test, y_val, y_test, scaler = preprocess_data(train_normal, val_df, test_df, target_column)

    # Create model
    input_dim = X_train.shape[1]
    model = Autoencoder(input_dim, hidden_dims, latent_dim)

    # Train model
    device = get_device()
    train_loader = create_dataloader(X_train, batch_size)
    criterion = getattr(torch.nn, criterion_name)()
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=learning_rate)

    train_losses = train_model(model, train_loader, criterion, optimizer, num_epochs, device)

    # Optimize threshold on validation set
    val_loader = create_dataloader(X_val, batch_size)
    val_errors = get_reconstruction_errors(model, val_loader, device)
    
    best_threshold = select_threshold(val_errors, y_val)
    print(f"Selected threshold: {best_threshold:.4f}")

    # Evaluate on test set
    test_loader = create_dataloader(X_test, batch_size)
    test_errors = get_reconstruction_errors(model, test_loader, device)

    metrics = evaluate_model(test_errors, y_test, best_threshold)

    # Save results
    results = {
        "model": "Autoencoder",
        "params": {
            "random_state": random_state,
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "latent_dim": latent_dim,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "criterion": criterion_name,
            "optimizer": optimizer_name
        },
        "best_threshold": float(best_threshold),
        "metrics": {
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1_score": float(metrics["f1_score"])
        },
    }

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    with open(reports_dir / "results.yaml", "w") as f:
        yaml.safe_dump(results, f)
    
    # Save mode, scaler, and threshold for future inference
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    torch.save(model.state_dict(), models_dir / "autoencoder.pth")

    with open(models_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(models_dir / "threshold.yaml", "w") as f:
        yaml.safe_dump({"threshold": float(best_threshold)}, f)


    # Plot and save training loss curve
    plot_training_loss(train_losses, reports_dir / "training_loss.png")


if __name__ == "__main__":
    main()