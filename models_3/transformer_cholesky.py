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

    Uses ROW-BASED TOKENIZATION: 32 tokens of 64 real values each.
    Each token represents one row of the density matrix (32 complex values).

    This captures physical structure:
    - Rows align with partial trace / computational basis structure
    - Attention between rows learns inter-row correlations (entanglement)
    - FFN handles within-row processing
    - 1000x fewer attention computations than element-wise tokenization

    ~411k parameters to match MLP.
    Output is guaranteed to be a valid density matrix.
    """

    def __init__(self, loss_fn=None):
        super().__init__()
        self.loss_fn = loss_fn

        # Row-based tokenization: 32 tokens, each is a row (32 complex = 64 real)
        self.seq_len = 32
        self.input_dim = 64  # 32 complex values = 64 real per row
        self.embed_dim = 64
        self.ffn_dim = 128
        self.num_heads = 4
        self.layers = 5  # Balanced depth for ~411k params
        self.cholesky_params = count_cholesky_params(32)  # 1024

        self.input_proj = nn.Linear(self.input_dim, self.embed_dim)
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

        self.down = nn.Linear(self.embed_dim, 32)
        self.up = nn.Linear(32, self.embed_dim)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ffn_dim,
            batch_first=True,
            activation="gelu",
            dropout=0.1,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=self.layers)

        # Output: each of 32 tokens produces 32 values -> 1024 Cholesky params
        self.output_proj = nn.Linear(self.embed_dim, 32)

        # Cholesky layer converts to valid density matrix
        self.cholesky = CholeskyDensityMatrix(32)

    def forward(self, x):
        B = x.shape[0]
        # x: (B, 2, 32, 32) -> row tokenization: (B, 32, 64)
        # Each row becomes a token: [real_row, imag_row] concatenated
        real_part = x[:, 0, :, :]  # (B, 32, 32)
        imag_part = x[:, 1, :, :]  # (B, 32, 32)
        x = torch.cat([real_part, imag_part], dim=-1)  # (B, 32, 64)

        x = self.input_proj(x) + self.pos_embedding  # (B, 32, embed_dim)
        enc = self.encoder(x)
        z = self.up(self.down(enc))
        dec = self.decoder(z, enc)

        # Each of 32 tokens outputs 32 values -> 1024 Cholesky params
        chol_params = self.output_proj(dec)  # (B, 32, 32)
        chol_params = chol_params.reshape(B, -1)  # (B, 1024)

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