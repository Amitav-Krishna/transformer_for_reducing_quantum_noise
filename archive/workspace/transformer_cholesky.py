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

    Uses ROW-BASED TOKENIZATION: 32 row tokens + 1 [CLS] token for global aggregation.
    Each row token represents one row of the density matrix (32 complex values).

    Architecture improvements (based on technical review):
    - [CLS] token aggregates global information across all rows
    - No bottleneck between encoder/decoder (prevents information destruction)
    - 8 attention heads for better diversity in attention patterns
    - Row-based tokenization: 1000x fewer attention ops than element-wise

    Output is guaranteed to be a valid density matrix via Cholesky decomposition.
    """

    def __init__(self, loss_fn=None):
        super().__init__()
        self.loss_fn = loss_fn

        # Row-based tokenization: 32 row tokens + 1 CLS token
        self.num_rows = 32
        self.seq_len = 33  # 32 rows + 1 CLS
        self.input_dim = 64  # 32 complex values = 64 real per row
        self.embed_dim = 64
        self.ffn_dim = 128
        self.num_heads = 8  # Increased from 4 for better attention diversity
        self.layers = 5
        self.cholesky_params = count_cholesky_params(32)  # 1024

        self.input_proj = nn.Linear(self.input_dim, self.embed_dim)

        # Learned [CLS] token for global aggregation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # Positional embeddings for 33 tokens (CLS + 32 rows)
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

        # No bottleneck - direct connection to decoder (prevents info destruction)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ffn_dim,
            batch_first=True,
            activation="gelu",
            dropout=0.1,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=self.layers)

        # Output: each of 32 row tokens produces 32 values -> 1024 Cholesky params
        # (CLS token output is discarded, it just helps with global aggregation)
        self.output_proj = nn.Linear(self.embed_dim, 32)

        # Cholesky layer converts to valid density matrix
        self.cholesky = CholeskyDensityMatrix(32)

    def forward(self, x):
        B = x.shape[0]
        # x: (B, 2, 32, 32) -> row tokenization: (B, 32, 64)
        # Each row becomes a token: [real_row, imag_row] concatenated
        real_part = x[:, 0, :, :]  # (B, 32, 32)
        imag_part = x[:, 1, :, :]  # (B, 32, 32)
        row_tokens = torch.cat([real_part, imag_part], dim=-1)  # (B, 32, 64)
        row_tokens = self.input_proj(row_tokens)  # (B, 32, embed_dim)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, row_tokens], dim=1)  # (B, 33, embed_dim)

        # Add positional embeddings
        x = x + self.pos_embedding  # (B, 33, embed_dim)

        # Encode
        enc = self.encoder(x)

        # Decode (no bottleneck - direct cross-attention to encoder output)
        dec = self.decoder(enc, enc)

        # Extract row tokens (skip CLS at position 0), project to Cholesky params
        row_output = dec[:, 1:, :]  # (B, 32, embed_dim) - skip CLS
        chol_params = self.output_proj(row_output)  # (B, 32, 32)
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