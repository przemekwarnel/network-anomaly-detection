from typing import List

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """Autoencoder model for network anomaly detection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        encoder_dims = [input_dim] + hidden_dims + [latent_dim]
        decoder_dims = [latent_dim] + hidden_dims[::-1] + [input_dim]

        encoder_layers = []
        for i in range(len(encoder_dims) - 1):
            encoder_layers.append(
                nn.Linear(encoder_dims[i], encoder_dims[i + 1])
            )
            if i < len(encoder_dims) - 2:
                encoder_layers.append(nn.ReLU())

        decoder_layers = []
        for i in range(len(decoder_dims) - 1):
            decoder_layers.append(
                nn.Linear(decoder_dims[i], decoder_dims[i+1])
            )
            if i < len(decoder_dims) - 2:
                decoder_layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass through the model."""

        x = self.encoder(x)
        x = self.decoder(x)

        return x
    