"""
Transformer Autoencoder with Cholesky output for density matrix denoising.

Same architecture as train_v8/transformer.py (~119k params) but outputs
1024 values via global pooling that are converted to a valid density matrix
via Cholesky parameterization.

Key difference from post-hoc version:
- Post-hoc: each token outputs 2 values → (B, 1024, 2) → reshape
- Cholesky: global pool → project to 1024 → Cholesky layer → valid density matrix
"""

import torch
import torch.nn as nn

from train_12.cholesky.cholesky_output import CholeskyDensityMatrix


class TransformerCholeskyAutoencoder(nn.Module):
    """
    Transformer autoencoder with Cholesky output layer.

    Architecture matches train_v8/transformer.py:
        - 1024 tokens, embed_dim=32, ffn_dim=64, 4 heads, 4 layers
        - Bottleneck: 32 -> 16 -> 32

    Output: global pool over tokens → project to 1024 → Cholesky → density matrix

    ~119k parameters (same as post-hoc projection version).
    """

    def __init__(self, loss_fn=None):
        super().__init__()
        self.loss_fn = loss_fn

        self.seq_len = 1024
        self.input_dim = 2
        # Same architecture as train_v8/transformer.py
        self.embed_dim = 32
        self.ffn_dim = 64
        self.num_heads = 4
        self.layers = 4

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

        # Global output: pool all tokens and project to 1024 Cholesky params
        # Using learned weighted pooling via a single projection
        self.global_pool = nn.Linear(self.embed_dim, 1)  # Attention weights per token
        self.output_proj = nn.Linear(self.embed_dim, 1024)

        # Cholesky output layer
        self.cholesky = CholeskyDensityMatrix(matrix_size=32)

    def forward(self, x):
        B = x.shape[0]
        # (B, 2, 32, 32) -> (B, 1024, 2)
        x = x.permute(0, 2, 3, 1).reshape(B, 1024, 2)

        x = self.input_proj(x) + self.pos_embedding
        enc = self.encoder(x)

        z = self.up(self.down(enc))
        dec = self.decoder(z, enc)  # (B, 1024, 32)

        # Global pooling: weighted average of all token embeddings
        attn_weights = torch.softmax(self.global_pool(dec), dim=1)  # (B, 1024, 1)
        pooled = (dec * attn_weights).sum(dim=1)  # (B, 32)

        # Project to Cholesky parameters
        cholesky_params = self.output_proj(pooled)  # (B, 1024)

        # Convert to valid density matrix
        out = self.cholesky(cholesky_params)  # (B, 2, 32, 32)

        return out

    def compute_loss(self, pred, target):
        return self.loss_fn(pred, target)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = TransformerCholeskyAutoencoder()
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

    # Check trace = 1
    traces = torch.diagonal(rho, dim1=-2, dim2=-1).sum(dim=-1)
    print(f"Traces: {traces.real}")

    # Check PSD
    eigvals = torch.linalg.eigvalsh(rho)
    print(f"Min eigenvalue: {eigvals.min().item():.6f} (should be >= 0)")
