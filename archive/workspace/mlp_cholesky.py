"""
MLP Autoencoder with Cholesky output for density matrix denoising.

Enforces physical validity (ρ ≥ 0, Tr(ρ) = 1) by construction via
Cholesky decomposition of the output layer.
"""

import torch
import torch.nn as nn
from models_3.cholesky_output import CholeskyDensityMatrix, count_cholesky_params


class MLPCholeskyAutoencoder(nn.Module):
    """
    MLP autoencoder with Cholesky output layer.

    Architecture:
        2048 -> 128 -> 142 -> 1024 -> Cholesky -> (2, 32, 32)

    ~427k parameters to match Transformer. Hidden sizes follow Morgillo et al.
    Dropout=0.1 matches Transformer regularization for fair comparison.
    Output is guaranteed to be a valid density matrix.
    """

    def __init__(self, loss_fn=None, dropout=0.1):
        super().__init__()
        self.loss_fn = loss_fn

        self.input_dim = 32 * 32 * 2  # 2048
        self.cholesky_params = count_cholesky_params(32)  # 1024

        # Encoder-decoder network with dropout matching Transformer
        # Second hidden layer slightly larger (142) to match Transformer params
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 142),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(142, self.cholesky_params),  # Output 1024 params for Cholesky
        )

        # Cholesky layer converts to valid density matrix
        self.cholesky = CholeskyDensityMatrix(32)

    def forward(self, x):
        B = x.shape[0]
        # Flatten: (B, 2, 32, 32) -> (B, 2048)
        x = x.reshape(B, -1)

        # Network produces Cholesky parameters
        chol_params = self.network(x)

        # Convert to valid density matrix
        rho = self.cholesky(chol_params)

        return rho

    def compute_loss(self, pred, target):
        return self.loss_fn(pred, target)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/amitav/codage/transformer_qnr')
    from losses.frob import FrobeniusFidelityLoss

    model = MLPCholeskyAutoencoder(loss_fn=FrobeniusFidelityLoss())
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

    # Check trace
    traces = torch.diagonal(rho, dim1=-2, dim2=-1).sum(dim=-1)
    print(f"Traces: {traces.real}")

    # Check PSD
    eigvals = torch.linalg.eigvalsh(rho)
    print(f"Min eigenvalue: {eigvals.min().item():.6f}")

    print(f"\nFor reference:")
    print(f"  Transformer (small): 119,506 params")