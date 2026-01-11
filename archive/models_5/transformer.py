"""
Transformer Autoencoder for density matrix denoising.

Uses ROW-BASED TOKENIZATION with PER-ROW OUTPUT.
Each row token outputs its own row - no global bottleneck.

Unconstrained output - use post-hoc projection for physical validity.
"""

import torch
import torch.nn as nn


class TransformerAutoencoder(nn.Module):
    """
    Transformer autoencoder with per-row output projection.

    Uses ROW-BASED TOKENIZATION: 32 row tokens.
    Each row token represents one row of the density matrix (32 complex values).
    Each row token outputs its corresponding row (64 values = 32 real + 32 imag).

    This is similar to the working v2 architecture but at row granularity
    instead of element granularity (32 tokens vs 1024 tokens).

    Output requires post-hoc projection to ensure valid density matrix.
    """

    def __init__(self, loss_fn=None):
        super().__init__()
        self.loss_fn = loss_fn

        # Row-based tokenization: 32 row tokens
        self.num_rows = 32
        self.row_dim = 64  # 32 complex values = 64 real per row
        self.embed_dim = 128
        self.ffn_dim = 256
        self.num_heads = 8
        self.layers = 4

        self.input_proj = nn.Linear(self.row_dim, self.embed_dim)

        # Positional embeddings for 32 row tokens
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.num_rows, self.embed_dim)
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

        # Bottleneck (50% compression, matching v2's ratio)
        self.down = nn.Linear(self.embed_dim, 64)
        self.up = nn.Linear(64, self.embed_dim)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ffn_dim,
            batch_first=True,
            activation="gelu",
            dropout=0.1,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=self.layers)

        # Per-row output projection: each row token outputs 64 values
        self.output_proj = nn.Linear(self.embed_dim, self.row_dim)

        # Initialize for stable training
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        # Positional embeddings from small normal (not zeros)
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
        # Scale output projection small for stable start
        nn.init.xavier_uniform_(self.output_proj.weight)
        self.output_proj.weight.data *= 0.1
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x):
        B = x.shape[0]
        # x: (B, 2, 32, 32) -> row tokenization: (B, 32, 64)
        # Each row becomes a token: [real_row, imag_row] concatenated
        real_part = x[:, 0, :, :]  # (B, 32, 32)
        imag_part = x[:, 1, :, :]  # (B, 32, 32)
        row_tokens = torch.cat([real_part, imag_part], dim=-1)  # (B, 32, 64)

        # Project to embedding dimension
        x = self.input_proj(row_tokens)  # (B, 32, embed_dim)

        # Add positional embeddings
        x = x + self.pos_embedding  # (B, 32, embed_dim)

        # Encode
        enc = self.encoder(x)  # (B, 32, embed_dim)

        # Bottleneck
        z = self.up(self.down(enc))  # (B, 32, embed_dim)

        # Decode with cross-attention to encoder
        dec = self.decoder(z, enc)  # (B, 32, embed_dim)

        # Per-row output projection
        out = self.output_proj(dec)  # (B, 32, 64)

        # Reshape to (B, 2, 32, 32)
        real_out = out[:, :, :32]  # (B, 32, 32)
        imag_out = out[:, :, 32:]  # (B, 32, 32)
        out = torch.stack([real_out, imag_out], dim=1)  # (B, 2, 32, 32)

        return out

    def compute_loss(self, pred, target):
        return self.loss_fn(pred, target)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/work/codage/transformer_qnr')
    from losses.frob import FrobeniusFidelityLoss

    model = TransformerAutoencoder(loss_fn=FrobeniusFidelityLoss())
    print(f"TransformerAutoencoder parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(4, 2, 32, 32)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
