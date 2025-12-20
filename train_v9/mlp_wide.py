"""
Wide MLP Autoencoder with 1024-dimensional bottleneck.

For comparison: tests whether transformer's advantage comes from
attention mechanism or just from having more latent dimensions.

v8 MLP: 2048 → 28 → 28 → 2048 (~117k params)
v9 Wide MLP: 2048 → 1024 → 1024 → 2048 (~5.2M params)
"""

import torch
import torch.nn as nn


class MLPWideAutoencoder(nn.Module):
    """
    Wide MLP autoencoder with 1024-dimensional hidden layer.

    Architecture:
        output = input + network(input)

    The network learns a CORRECTION rather than full reconstruction.
    Uses 1024 hidden dimensions to match transformer's effective latent space.

    ~5.2M parameters.
    """

    def __init__(self, loss_fn=None, hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.loss_fn = loss_fn

        self.input_dim = 32 * 32 * 2  # 2048
        self.hidden_dim = hidden_dim

        # Correction network with wide hidden layer
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.input_dim),
        )

        # Initialize output layer small so initial output ≈ input
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, x):
        B = x.shape[0]
        # Flatten: (B, 2, 32, 32) -> (B, 2048)
        x_flat = x.reshape(B, -1)

        # Learn correction
        correction = self.network(x_flat)

        # Output = input + correction (residual)
        out = x_flat + correction

        # Reshape to (B, 2, 32, 32)
        out = out.reshape(B, 2, 32, 32)

        return out

    def compute_loss(self, pred, target):
        return self.loss_fn(pred, target)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = MLPWideAutoencoder()
    print(f"MLPWideAutoencoder parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(4, 2, 32, 32)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")

    # Check initial output ≈ input
    diff = (y - x).abs().mean()
    print(f"Initial output-input diff: {diff:.6f} (should be ~0)")
