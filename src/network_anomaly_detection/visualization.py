import matplotlib.pyplot as plt
from pathlib import Path


def plot_training_loss(train_losses: list, path: Path | str):
    """Plot and save the training loss curve."""

    fig, ax = plt.subplots()
    ax.plot(train_losses, label="Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)