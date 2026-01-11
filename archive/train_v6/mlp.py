"""
MLP Autoencoder for density matrix denoising.

~120k parameters to match v2 Transformer (119,506 params).
"""

import torch
import torch.nn as nn


class MLPAutoencoder(nn.Module):
    """
    MLP autoencoder with ~120k parameters.

    Architecture:
        2048 -> 28 -> 14 -> 28 -> 2048

    Parameter count: ~117,590 (close to Transformer's 119,506)
    """

    def __init__(self, loss_fn=None, dropout=0.1):
        super().__init__()
        self.loss_fn = loss_fn

        self.input_dim = 32 * 32 * 2  # 2048
        self.output_dim = 32 * 32 * 2  # 2048

        # Small network to match Transformer param count
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 28),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(28, 14),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(14, 28),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(28, self.output_dim),
        )

    def forward(self, x):
        B = x.shape[0]
        # Flatten: (B, 2, 32, 32) -> (B, 2048)
        x = x.reshape(B, -1)

        # Network produces output
        out = self.network(x)

        # Reshape to (B, 2, 32, 32)
        out = out.reshape(B, 2, 32, 32)

        return out

    def compute_loss(self, pred, target):
        return self.loss_fn(pred, target)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = MLPAutoencoder()
    print(f"MLPAutoencoder parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(4, 2, 32, 32)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
