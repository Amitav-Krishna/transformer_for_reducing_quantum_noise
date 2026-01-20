"""
Hierarchical MLP for 5-qubit density matrices (float64).

Control experiment: Same patch tokenization as Transformer, but uses
MLP-Mixer style token mixing instead of attention.

Specs:
- Matrix size: 32x32
- Patch size: 4x4
- Tokens: 64
- Params: ~1,092,960 (matched to Transformer)
- Layers: 4 encoder + 4 decoder
- Token hidden: 384
- Channel hidden: 320
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse patch embed/unembed from transformer (local import)
from train_18.models.transformer_5qubit import PatchEmbed, PatchUnembed


class MLPMixerBlock(nn.Module):
    """
    MLP-Mixer style block: token mixing + channel mixing.

    Replaces attention with two MLPs:
    1. Token mixing: mix information across patches (replaces attention)
    2. Channel mixing: process each patch independently (same as FFN)
    """

    def __init__(
        self,
        num_patches,
        embed_dim,
        token_hidden_dim,
        channel_hidden_dim,
        dropout=0.1,
        dtype=torch.float64,
    ):
        super().__init__()

        # Token mixing (across patches) - replaces attention
        self.norm1 = nn.LayerNorm(embed_dim, dtype=dtype)
        self.token_mixing = nn.Sequential(
            nn.Linear(num_patches, token_hidden_dim, dtype=dtype),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_hidden_dim, num_patches, dtype=dtype),
            nn.Dropout(dropout),
        )

        # Channel mixing (per patch) - same as FFN
        self.norm2 = nn.LayerNorm(embed_dim, dtype=dtype)
        self.channel_mixing = nn.Sequential(
            nn.Linear(embed_dim, channel_hidden_dim, dtype=dtype),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channel_hidden_dim, embed_dim, dtype=dtype),
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
    - Patch embedding: 32x32 -> 64 tokens (4x4 patches)
    - MLP-Mixer encoder: 4 layers
    - Bottleneck: embed_dim -> embed_dim/2 -> embed_dim
    - MLP-Mixer decoder: 4 layers
    - Patch unembed: 64 tokens -> 32x32

    Parameters: ~1,092,960 (matched to Transformer)
    """

    def __init__(
        self,
        loss_fn=None,
        embed_dim=128,
        token_hidden_dim=384,
        channel_hidden_dim=320,
        num_layers=4,
        dtype=torch.float64,
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.dtype = dtype
        self.num_patches = 64

        # Patch embedding (same as Transformer)
        self.patch_embed = PatchEmbed(
            matrix_size=32,
            patch_size=4,
            in_channels=2,
            embed_dim=embed_dim,
            dtype=dtype,
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, embed_dim, dtype=dtype))

        # Encoder
        self.encoder = nn.ModuleList(
            [
                MLPMixerBlock(
                    self.num_patches,
                    embed_dim,
                    token_hidden_dim,
                    channel_hidden_dim,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        # Bottleneck
        self.down = nn.Linear(embed_dim, embed_dim // 2, dtype=dtype)
        self.up = nn.Linear(embed_dim // 2, embed_dim, dtype=dtype)

        # Decoder
        self.decoder = nn.ModuleList(
            [
                MLPMixerBlock(
                    self.num_patches,
                    embed_dim,
                    token_hidden_dim,
                    channel_hidden_dim,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        # Output projection
        self.patch_unembed = PatchUnembed(
            matrix_size=32,
            patch_size=4,
            out_channels=2,
            embed_dim=embed_dim,
            dtype=dtype,
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
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
        if self.loss_fn is None:
            raise ValueError("loss_fn not set")
        return self.loss_fn(pred, target)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = HierarchicalMLP5Qubit()
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Model dtype: {next(model.parameters()).dtype}")

    x = torch.randn(4, 2, 32, 32, dtype=torch.float64)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}, Dtype: {y.dtype}")
    assert y.shape == (4, 2, 32, 32)
    assert y.dtype == torch.float64
    print("OK")
