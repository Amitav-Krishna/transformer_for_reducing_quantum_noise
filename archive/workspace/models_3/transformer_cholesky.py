"""
Transformer Autoencoder with Cholesky output for density matrix denoising.

Enforces physical validity (ρ ≥ 0, Tr(ρ) = 1) by construction via
Cholesky decomposition of the output layer.
"""

import torch
import torch.nn as nn
from models_3.cholesky_output import CholeskyDensityMatrix, count_cholesky_params


class TransformerCholeskyAutoencoder(nn.Module):
    """
    Transformer autoencoder with Cholesky output layer.

    Architecture matches the original small transformer (~119k params)
    but outputs Cholesky parameters instead of raw density matrix values.

    Output is guaranteed to be a valid density matrix.
    """

    def __init__(self, loss_fn=None):
        super().__init__()
        self.loss_fn = loss_fn

        self.seq_len = 1024
        self.input_dim = 2
        self.embed_dim = 32
        self.ffn_dim = 64
        self.num_heads = 4
        self.layers = 4
        self.cholesky_params = count_cholesky_params(32)  # 1024

        self.input_proj = nn.Linear(2, self.embed_dim)
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.seq_len, self.embed_dim)
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ffn_dim,
            batch_first=True,
            activation="gelu",
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.layers)

        self.down = nn.Linear(self.embed_dim, 16)
        self.up = nn.Linear(16, self.embed_dim)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ffn_dim,
            batch_first=True,
            activation="gelu",
            dropout=0.1,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=self.layers)

        # Output: project each token to 1 value, giving 1024 Cholesky params
        self.output_proj = nn.Linear(self.embed_dim, 1)

        # Cholesky layer converts to valid density matrix
        self.cholesky = CholeskyDensityMatrix(32)

    def forward(self, x):
        B = x.shape[0]
        x = x.permute(0, 2, 3, 1).reshape(B, 1024, 2)  # (B, 1024, 2)
        x = self.input_proj(x) + self.pos_embedding
        enc = self.encoder(x)
        z = self.up(self.down(enc))
        dec = self.decoder(z, enc)

        # Project each token to 1 value -> 1024 Cholesky parameters
        chol_params = self.output_proj(dec).squeeze(-1)  # (B, 1024)

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

    model = TransformerCholeskyAutoencoder(loss_fn=FrobeniusFidelityLoss())
    print(f"TransformerCholeskyAutoencoder parameters: {count_parameters(model):,}")

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
    print(f"  Original Transformer: 119,506 params")