"""
Transformer Autoencoder for density matrix denoising.

This is the v2 architecture that works well.
Uses ELEMENT-WISE TOKENIZATION: 1024 tokens, each representing one matrix element.

~119,506 parameters.
"""

import torch
import torch.nn as nn


class TransformerAutoencoder(nn.Module):
    """
    Transformer autoencoder with element-wise tokenization.

    Each of the 1024 matrix elements becomes a token with 2 values (real, imag).
    This allows attention to capture element-to-element correlations directly.

    Architecture:
        - 1024 tokens, embed_dim=32, ffn_dim=64, 4 heads, 4 layers
        - Bottleneck: 32 -> 16 -> 32 (50% compression)
        - Per-token output: each token outputs 2 values

    Parameters: ~119,506
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

        self.output_proj = nn.Linear(self.embed_dim, 2)

    def forward(self, x):
        B = x.shape[0]
        # (B, 2, 32, 32) -> (B, 1024, 2)
        x = x.permute(0, 2, 3, 1).reshape(B, 1024, 2)

        x = self.input_proj(x) + self.pos_embedding
        enc = self.encoder(x)

        z = self.up(self.down(enc))
        dec = self.decoder(z, enc)

        out = self.output_proj(dec)        # (B, 1024, 2)
        out = out.reshape(B, 32, 32, 2)    # (B, 32, 32, 2)
        out = out.permute(0, 3, 1, 2)      # (B, 2, 32, 32)
        return out

    def compute_loss(self, pred, target):
        return self.loss_fn(pred, target)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = TransformerAutoencoder()
    print(f"TransformerAutoencoder parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(4, 2, 32, 32)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
