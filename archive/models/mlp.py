"""
MLP Autoencoder baseline for density matrix denoising.

Architecture inspired by Morgillo et al. (arXiv:2309.11949) who use
feedforward neural networks with ReLU activations for quantum state
reconstruction. We constrain to ~119k parameters to match the transformer.

This serves as a baseline to test whether the transformer's advantage
comes from attention specifically, or just from having a global receptive
field. Both MLP and Transformer have global receptive fields.
"""

import torch
import torch.nn as nn


class MLPAutoencoder(nn.Module):
    """
    MLP autoencoder with ~119k parameters (matches small transformer).

    Architecture:
        2048 -> 28 -> 28 -> 2048

    Key design choices from Morgillo et al.:
    - ReLU activation
    - Simple feedforward structure
    - No dropout

    Parameter count constrained to match transformer for fair comparison.
    """

    def __init__(self, loss_fn=None):
        super().__init__()
        self.loss_fn = loss_fn

        self.input_dim = 32 * 32 * 2  # 2048

        # Constrained to ~119k params to match transformer
        # With 2048 input, hidden layers must be small
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 28),
            nn.ReLU(),
            nn.Linear(28, 28),
            nn.ReLU(),
            nn.Linear(28, self.input_dim),
        )

    def forward(self, x):
        B = x.shape[0]
        # Flatten: (B, 2, 32, 32) -> (B, 2048)
        x = x.reshape(B, -1)

        out = self.network(x)

        # Reshape: (B, 2048) -> (B, 2, 32, 32)
        out = out.reshape(B, 2, 32, 32)
        return out

    def compute_loss(self, pred, target):
        return self.loss_fn(pred, target)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from losses.frob import FrobeniusFidelityLoss

    model = MLPAutoencoder(loss_fn=FrobeniusFidelityLoss())
    print(f"MLPAutoencoder parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(4, 2, 32, 32)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")

    # Compare to transformer
    print(f"\nFor reference:")
    print(f"  Transformer (small): 119,506 params")