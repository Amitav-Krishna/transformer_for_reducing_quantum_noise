"""
Wide Hierarchical MLP for 5-qubit density matrices.

Capacity control experiment: Tests whether the hierarchical Transformer's
advantage comes from attention or just from having more parameters.

Comparison:
- Hierarchical Transformer (5-qubit): ~1.1M params
- Hierarchical MLP (5-qubit): ~1.1M params (matched)
- Hierarchical MLP Wide (5-qubit): ~5M params (capacity control)
- v9 Flat Wide MLP: ~5.2M params (original capacity control)

If Wide MLP still underperforms Transformer, architecture > capacity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reuse patch embedding/unembedding from transformer
from models.transformer_hierarchical_5qubit import PatchEmbed, PatchUnembed


class MLPMixerBlockWide(nn.Module):
    """
    Wide MLP-Mixer block with larger hidden dimensions.
    """

    def __init__(
        self, num_patches, embed_dim, token_hidden_dim, channel_hidden_dim, dropout=0.1
    ):
        super().__init__()

        # Token mixing (across patches)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.token_mixing = nn.Sequential(
            nn.Linear(num_patches, token_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_hidden_dim, num_patches),
            nn.Dropout(dropout),
        )

        # Channel mixing (per patch)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.channel_mixing = nn.Sequential(
            nn.Linear(embed_dim, channel_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channel_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, num_patches, embed_dim)

        # Token mixing
        x_norm = self.norm1(x)
        x_t = x_norm.transpose(1, 2)
        x_t = self.token_mixing(x_t)
        x = x + x_t.transpose(1, 2)

        # Channel mixing
        x = x + self.channel_mixing(self.norm2(x))

        return x


class HierarchicalMLP5QubitWide(nn.Module):
    """
    Wide Hierarchical MLP for 5-qubit density matrix denoising.

    Architecture:
    - Patch embedding: 32×32 → 64 tokens (4×4 patches)
    - Wide MLP-Mixer encoder-decoder: 6 layers each (vs 4 for matched)
    - Larger embed_dim: 256 (vs 128)
    - Larger hidden dims
    - Patch unembed: 64 tokens → 32×32

    Target: ~5M parameters (matching v9 flat wide MLP)
    """

    def __init__(
        self,
        loss_fn=None,
        embed_dim=192,  # 1.5x larger than matched version
        token_hidden_dim=256,  # Token mixing hidden dim
        channel_hidden_dim=1024,  # Large channel mixing
        num_layers=6,  # More layers
    ):
        # Tuned to give ~5.2M params, matching v9 flat wide MLP
        super().__init__()
        self.loss_fn = loss_fn
        self.num_patches = 64

        # Patch embedding with larger embed_dim
        self.patch_embed = PatchEmbed(
            matrix_size=32, patch_size=4, in_channels=2, embed_dim=embed_dim
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, embed_dim))

        # Encoder (more and wider layers)
        self.encoder = nn.ModuleList(
            [
                MLPMixerBlockWide(
                    self.num_patches, embed_dim, token_hidden_dim, channel_hidden_dim
                )
                for _ in range(num_layers)
            ]
        )

        # Bottleneck
        self.down = nn.Linear(embed_dim, embed_dim // 2)
        self.up = nn.Linear(embed_dim // 2, embed_dim)

        # Decoder
        self.decoder = nn.ModuleList(
            [
                MLPMixerBlockWide(
                    self.num_patches, embed_dim, token_hidden_dim, channel_hidden_dim
                )
                for _ in range(num_layers)
            ]
        )

        # Output projection with larger embed_dim
        self.patch_unembed = PatchUnembed(
            matrix_size=32, patch_size=4, out_channels=2, embed_dim=embed_dim
        )

        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: (B, 2, 32, 32)

        tokens = self.patch_embed(x)
        tokens = tokens + self.pos_embed

        for block in self.encoder:
            tokens = block(tokens)

        z = self.up(F.gelu(self.down(tokens)))

        for block in self.decoder:
            z = block(z)

        out = self.patch_unembed(z)

        return out

    def compute_loss(self, pred, target):
        return self.loss_fn(pred, target)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from losses.frob import FrobeniusFidelityLoss
    from models.transformer_hierarchical_5qubit import HierarchicalTransformer5Qubit
    from train_15.mlp_hierarchical_5qubit import HierarchicalMLP5Qubit
    from train_v9.mlp_wide import MLPWideAutoencoder

    # Create all models for comparison
    transformer = HierarchicalTransformer5Qubit(loss_fn=FrobeniusFidelityLoss())
    mlp_matched = HierarchicalMLP5Qubit(loss_fn=FrobeniusFidelityLoss())
    mlp_wide = HierarchicalMLP5QubitWide(loss_fn=FrobeniusFidelityLoss())
    flat_wide = MLPWideAutoencoder(loss_fn=FrobeniusFidelityLoss())

    print("=== Parameter Comparison ===")
    print(f"Hierarchical Transformer: {count_parameters(transformer):,}")
    print(f"Hierarchical MLP (matched): {count_parameters(mlp_matched):,}")
    print(f"Hierarchical MLP Wide: {count_parameters(mlp_wide):,}")
    print(f"Flat Wide MLP (v9): {count_parameters(flat_wide):,}")

    # Test forward pass
    x = torch.randn(4, 2, 32, 32)
    y = mlp_wide(x)
    print(f"\nInput: {x.shape}, Output: {y.shape}")
    assert y.shape == (4, 2, 32, 32)
    print("Forward pass successful!")
