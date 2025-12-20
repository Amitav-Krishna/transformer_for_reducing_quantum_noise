"""
Transformer for Pauli basis representation of density matrices.

Input: (B, 1024) real-valued Pauli coefficients
Output: (B, 1024) predicted Pauli coefficients for denoised state

This tests whether the Transformer's advantage persists when the
"spatial" structure of the matrix is replaced with basis coefficients.
In Pauli basis, there is no inherent spatial locality like in the 32×32
matrix representation, so we expect Transformer advantage to be smaller.
"""

import torch
import torch.nn as nn


class TransformerPauliAutoencoder(nn.Module):
    """
    Transformer autoencoder for Pauli basis representation.

    Input: (B, 1024) Pauli coefficients
    Each coefficient becomes a token with embedding.

    Architecture similar to original Transformer but adapted for
    1024-dimensional input (vs 1024-element 2D matrix).
    """

    def __init__(self, loss_fn=None):
        super().__init__()
        self.loss_fn = loss_fn

        self.n_tokens = 1024  # One token per Pauli coefficient
        self.embed_dim = 32   # Embedding dimension (same as original)
        self.ff_dim = 64      # Feed-forward dimension

        # Each token is a scalar (Pauli coefficient) → embed to embed_dim
        self.token_embed = nn.Linear(1, self.embed_dim)

        # Encoder: 4 transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=4,
            dim_feedforward=self.ff_dim,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Bottleneck per-token compression (same as original)
        self.bottleneck = nn.Sequential(
            nn.Linear(self.embed_dim, 16),
            nn.ReLU(),
            nn.Linear(16, self.embed_dim)
        )

        # Decoder: 4 transformer layers (mirroring encoder)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=4,
            dim_feedforward=self.ff_dim,
            batch_first=True,
            activation='gelu'
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=4)

        # Output projection: token → scalar Pauli coefficient
        self.output_proj = nn.Linear(self.embed_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (B, 1024) Pauli coefficients

        Returns:
            out: (B, 1024) predicted Pauli coefficients
        """
        B = x.shape[0]

        # Embed each Pauli coefficient
        # x is (B, 1024) → unsqueeze to (B, 1024, 1) → embed to (B, 1024, embed_dim)
        x_embed = self.token_embed(x.unsqueeze(-1))  # (B, 1024, embed_dim)

        # Encoder
        h = self.encoder(x_embed)  # (B, 1024, embed_dim)

        # Per-token bottleneck
        h_bottleneck = torch.stack([
            self.bottleneck(h[:, i, :]) for i in range(self.n_tokens)
        ], dim=1)  # (B, 1024, embed_dim)

        # Decoder
        h_decoded = self.decoder(h_bottleneck)  # (B, 1024, embed_dim)

        # Output projection: (B, 1024, embed_dim) → (B, 1024, 1) → (B, 1024)
        out = self.output_proj(h_decoded).squeeze(-1)  # (B, 1024)

        return out

    def compute_loss(self, pred, target):
        return self.loss_fn(pred, target)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = TransformerPauliAutoencoder()
    params = count_parameters(model)
    print(f"TransformerPauliAutoencoder parameters: {params:,}")

    # Test forward pass
    x = torch.randn(4, 1024)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
