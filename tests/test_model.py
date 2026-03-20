import torch

from network_anomaly_detection.model import Autoencoder


def test_autoencoder():
    input_dim = 20
    hidden_dims = [10, 5]
    latent_dim = 2

    model = Autoencoder(input_dim, hidden_dims, latent_dim)

    X_dummy = torch.randn(4, input_dim)
    output = model(X_dummy)

    assert isinstance(output, torch.Tensor)
    assert output.shape == X_dummy.shape