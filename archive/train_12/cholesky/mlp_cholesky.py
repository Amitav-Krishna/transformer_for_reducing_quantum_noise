"""
MLP Autoencoder with Cholesky output for density matrix denoising.

Same architecture as train_v8/mlp.py (~117k params) but outputs 1024 values
that are converted to a valid density matrix via Cholesky parameterization.

This ensures physical validity by construction (PSD, trace=1, Hermitian).
"""

import torch
import torch.nn as nn

from train_12.cholesky.cholesky_output import CholeskyDensityMatrix


class MLPCholeskyAutoencoder(nn.Module):
    """
    MLP autoencoder with Cholesky output layer.

    Architecture matches train_v8/mlp.py:
        - Input: (B, 2, 32, 32) density matrix
        - Hidden dims: 2048 → 28 → 28 → 1024
        - Cholesky layer: 1024 → valid density matrix

    ~117k parameters (same as post-hoc projection version).
    """

    def __init__(self, loss_fn=None, dropout=0.1):
        super().__init__()
        self.loss_fn = loss_fn

        self.input_dim = 32 * 32 * 2  # 2048
        self.cholesky_params = 1024   # Params needed for Cholesky

        # Same architecture as train_v8/mlp.py (hidden=28), just output to 1024
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 28),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(28, 28),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(28, self.cholesky_params),
        )

        # Cholesky output layer
        self.cholesky = CholeskyDensityMatrix(matrix_size=32)

    def forward(self, x):
        B = x.shape[0]
        # Flatten: (B, 2, 32, 32) -> (B, 2048)
        x_flat = x.reshape(B, -1)

        # Get Cholesky parameters
        cholesky_params = self.network(x_flat)  # (B, 1024)

        # Convert to valid density matrix
        out = self.cholesky(cholesky_params)  # (B, 2, 32, 32)

        return out

    def compute_loss(self, pred, target):
        return self.loss_fn(pred, target)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = MLPCholeskyAutoencoder()
    print(f"MLPCholeskyAutoencoder parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(4, 2, 32, 32)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")

    # Verify output is valid density matrix
    rho = y[:, 0] + 1j * y[:, 1]

    # Check Hermitian
    is_hermitian = torch.allclose(rho, rho.conj().transpose(-2, -1), atol=1e-5)
    print(f"Is Hermitian: {is_hermitian}")

    # Check trace = 1
    traces = torch.diagonal(rho, dim1=-2, dim2=-1).sum(dim=-1)
    print(f"Traces: {traces.real}")

    # Check PSD
    eigvals = torch.linalg.eigvalsh(rho)
    print(f"Min eigenvalue: {eigvals.min().item():.6f} (should be >= 0)")
