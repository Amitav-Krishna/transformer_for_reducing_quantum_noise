"""
Hierarchical MLP for 5-qubit density matrices.

Control experiment: Uses identical patch embedding as the hierarchical Transformer,
but replaces attention with MLP-based mixing. This isolates the comparison between
attention and feedforward mechanisms.

Architecture comparison:
- Transformer: patch_embed -> attention blocks -> patch_unembed
- MLP: patch_embed -> MLP mixer blocks -> patch_unembed

Both see the same 64-token representation (4×4 patches from 32×32 matrix).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reuse patch embedding/unembedding from transformer
from models.transformer_hierarchical_5qubit import PatchEmbed, PatchUnembed


class MLPMixerBlock(nn.Module):
    """
    MLP-Mixer style block: token mixing + channel mixing.

    Replaces attention with two MLPs:
    1. Token mixing: mix information across patches (replaces attention)
    2. Channel mixing: process each patch independently (same as FFN)
    """

    def __init__(
        self, num_patches, embed_dim, token_hidden_dim, channel_hidden_dim, dropout=0.1
    ):
        super().__init__()

        # Token mixing (across patches) - replaces attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.token_mixing = nn.Sequential(
            nn.Linear(num_patches, token_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_hidden_dim, num_patches),
            nn.Dropout(dropout),
        )

        # Channel mixing (per patch) - same as FFN
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

        # Token mixing (transpose to mix across patches)
        x_norm = self.norm1(x)
        x_t = x_norm.transpose(1, 2)  # (B, embed_dim, num_patches)
        x_t = self.token_mixing(x_t)  # (B, embed_dim, num_patches)
        x = x + x_t.transpose(1, 2)  # (B, num_patches, embed_dim)

        # Channel mixing
        x = x + self.channel_mixing(self.norm2(x))

        return x


class HierarchicalMLP5Qubit(nn.Module):
    """
    Hierarchical MLP for 5-qubit density matrix denoising.

    Architecture:
    - Patch embedding: 32×32 → 64 tokens (4×4 patches) [SHARED with Transformer]
    - MLP-Mixer encoder-decoder: 4 layers each
    - Bottleneck: embed_dim → embed_dim/2 → embed_dim
    - Patch unembed: 64 tokens → 32×32 [SHARED with Transformer]

    Key difference from Transformer:
    - Token mixing via MLP instead of attention
    - O(num_patches) complexity instead of O(num_patches²)
    - Cannot learn input-dependent mixing patterns

    Parameters matched to Transformer (~1.1M) for fair comparison.
    """

    def __init__(
        self,
        loss_fn=None,
        embed_dim=128,
        token_hidden_dim=384,  # Hidden dim for token mixing MLP
        channel_hidden_dim=320,  # Hidden dim for channel mixing
        num_layers=4,
    ):
        # Note: token_hidden=384, channel_hidden=320 matches parameter count
        # with HierarchicalTransformer5Qubit
        super().__init__()
        self.loss_fn = loss_fn
        self.num_patches = 64  # 8×8 grid of 4×4 patches

        # Patch embedding (SAME as Transformer)
        self.patch_embed = PatchEmbed(
            matrix_size=32, patch_size=4, in_channels=2, embed_dim=embed_dim
        )

        # Positional embedding for 64 patches
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, embed_dim))

        # Encoder (MLP-Mixer blocks instead of Transformer blocks)
        self.encoder = nn.ModuleList(
            [
                MLPMixerBlock(
                    self.num_patches, embed_dim, token_hidden_dim, channel_hidden_dim
                )
                for _ in range(num_layers)
            ]
        )

        # Bottleneck (SAME as Transformer)
        self.down = nn.Linear(embed_dim, embed_dim // 2)
        self.up = nn.Linear(embed_dim // 2, embed_dim)

        # Decoder (MLP-Mixer blocks)
        self.decoder = nn.ModuleList(
            [
                MLPMixerBlock(
                    self.num_patches, embed_dim, token_hidden_dim, channel_hidden_dim
                )
                for _ in range(num_layers)
            ]
        )

        # Output projection (SAME as Transformer)
        self.patch_unembed = PatchUnembed(
            matrix_size=32, patch_size=4, out_channels=2, embed_dim=embed_dim
        )

        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: (B, 2, 32, 32)

        # Patch embed (SAME as Transformer)
        tokens = self.patch_embed(x)  # (B, 64, embed_dim)
        tokens = tokens + self.pos_embed

        # Encoder
        for block in self.encoder:
            tokens = block(tokens)

        # Bottleneck (SAME as Transformer)
        z = self.up(F.gelu(self.down(tokens)))

        # Decoder
        for block in self.decoder:
            z = block(z)

        # Unembed (SAME as Transformer)
        out = self.patch_unembed(z)  # (B, 2, 32, 32)

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

    # Create both models
    mlp_model = HierarchicalMLP5Qubit(loss_fn=FrobeniusFidelityLoss())
    transformer_model = HierarchicalTransformer5Qubit(loss_fn=FrobeniusFidelityLoss())

    mlp_params = count_parameters(mlp_model)
    transformer_params = count_parameters(transformer_model)

    print(f"HierarchicalMLP5Qubit parameters: {mlp_params:,}")
    print(f"HierarchicalTransformer5Qubit parameters: {transformer_params:,}")
    print(f"Ratio (MLP/Transformer): {mlp_params / transformer_params:.2f}x")

    # Test forward pass
    x = torch.randn(4, 2, 32, 32)
    y = mlp_model(x)
    print(f"\nInput: {x.shape}, Output: {y.shape}")

    # Verify shapes
    assert y.shape == (4, 2, 32, 32), f"Expected (4, 2, 32, 32), got {y.shape}"
    print("Forward pass successful!")
