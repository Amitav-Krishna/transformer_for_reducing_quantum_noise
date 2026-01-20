"""
CNN Autoencoder for density matrix denoising.

Scaled-down version of models/cnn.py to match ~120k parameters
of the MLP and Transformer baselines.

Architecture:
    - Encoder: 3 conv layers with MaxPool (2→19→38→76 channels)
    - Bottleneck: 76 channels at 4×4 spatial resolution
    - Decoder: 3 deconv layers with Upsample

~118k parameters.
"""

import torch
import torch.nn as nn


class CNNAutoencoder(nn.Module):
    """
    CNN autoencoder with ~118k parameters.

    Scaled down from original (748k) to match MLP/Transformer capacity.
    Channel progression: 2 → 19 → 38 → 76 → 76 → 38 → 19 → 2
    """

    def __init__(self, loss_fn=None):
        super().__init__()
        self.loss_fn = loss_fn

        # Encoder: downsample 32→16→8→4
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 19, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32→16

            nn.Conv2d(19, 38, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16→8

            nn.Conv2d(38, 76, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 8→4
        )

        # Bottleneck at 4×4 resolution
        self.bottleneck = nn.Sequential(
            nn.Conv2d(76, 76, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Decoder: upsample 4→8→16→32
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),  # 4→8
            nn.ConvTranspose2d(76, 38, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode="nearest"),  # 8→16
            nn.ConvTranspose2d(38, 19, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode="nearest"),  # 16→32
            nn.ConvTranspose2d(19, 2, kernel_size=3, padding=1),
        )

    def forward(self, x):
        z = self.bottleneck(self.encoder(x))
        return self.decoder(z)

    def compute_loss(self, pred, target):
        return self.loss_fn(pred, target)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = CNNAutoencoder()
    print(f"CNNAutoencoder parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(4, 2, 32, 32)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")

    # Verify output shape matches input
    assert x.shape == y.shape, f"Shape mismatch: {x.shape} vs {y.shape}"
    print("Shape verification passed!")
