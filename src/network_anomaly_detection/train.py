import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def create_dataloader(X: pd.DataFrame, batch_size: int, shuffle: bool = False) -> DataLoader:
    """Create a DataLoader for the given dataset."""

    dataset = TensorDataset(torch.tensor(X.to_numpy(), dtype=torch.float32))

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_device() -> torch.device:
    """Get the available device (GPU or CPU)."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, train_loader, criterion, optimizer, num_epochs: int, device: torch.device) -> list[float]:
    """Train the autoencoder model."""

    model = model.to(device)

    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            inputs = batch[0].to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return train_losses