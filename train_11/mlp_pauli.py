"""
MLP for Pauli basis representation of density matrices.

Input: (B, 1024) real-valued Pauli coefficients
Output: (B, 1024) predicted Pauli coefficients for denoised state

Same architecture as regular MLP but takes 1024-dimensional Pauli input
instead of 2048-dimensional flattened (real/imag) input.
"""

import torch
import torch.nn as nn


class MLPPauliAutoencoder(nn.Module):
    """
    MLP autoencoder for Pauli basis representation.

    Input shape: (B, 1024) where each element is a Pauli coefficient
    Output shape: (B, 1024) predicted Pauli coefficients

    Architecture: 1024 -> 256 -> 256 -> 1024 (similar bottleneck ratio)
    """

    def __init__(self, loss_fn=None):
        super().__init__()
        self.loss_fn = loss_fn

        self.input_dim = 4 ** 5  # 1024 Pauli coefficients for 5 qubits

        # Similar bottleneck ratio to original MLP (2048 -> 28)
        # Original ratio: 28/2048 = 1.37%
        # New bottleneck: 14 (1024 * 1.37% ≈ 14)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 14),
            nn.ReLU(),
            nn.Linear(14, 128),
            nn.ReLU(),
            nn.Linear(128, self.input_dim),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 1024) Pauli coefficients

        Returns:
            out: (B, 1024) predicted Pauli coefficients
        """
        return self.network(x)

    def compute_loss(self, pred, target):
        return self.loss_fn(pred, target)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = MLPPauliAutoencoder()
    params = count_parameters(model)
    print(f"MLPPauliAutoencoder parameters: {params:,}")

    # Test forward pass
    x = torch.randn(4, 1024)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
