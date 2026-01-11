"""
MLP Autoencoder with residual connection for density matrix denoising.

Instead of reconstructing from scratch, learns a correction to the input.
~120k parameters to match Transformer.
"""

import torch
import torch.nn as nn


class MLPResidualAutoencoder(nn.Module):
    """
    MLP autoencoder with residual connection.

    Architecture:
        output = input + network(input)

    The network learns a CORRECTION rather than full reconstruction.
    This preserves input information and only modifies what's needed.

    ~120k parameters.
    """

    def __init__(self, loss_fn=None, dropout=0.1):
        super().__init__()
        self.loss_fn = loss_fn

        self.input_dim = 32 * 32 * 2  # 2048

        # Correction network: learns what to ADD to input
        # Hidden dim 28 gives ~117k params (matching original MLP)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 28),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(28, 28),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(28, self.input_dim),
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
    model = MLPResidualAutoencoder()
    print(f"MLPResidualAutoencoder parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(4, 2, 32, 32)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")

    # Check initial output ≈ input
    diff = (y - x).abs().mean()
    print(f"Initial output-input diff: {diff:.6f} (should be ~0)")
