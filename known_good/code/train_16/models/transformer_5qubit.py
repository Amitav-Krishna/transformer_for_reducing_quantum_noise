"""
Hierarchical Transformer for 5-qubit density matrices (float64).

Reuses architecture from models/transformer_hierarchical_5qubit.py.
Float64 conversion happens at model initialization in training scripts.

Specs:
- Matrix size: 32x32
- Patch size: 4x4
- Tokens: 64
- Params: ~1,092,960
- Layers: 4 encoder + 4 decoder
- Heads: 8
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """Convert 32x32 matrix to 64 patch tokens (4x4 patches)."""

    def __init__(
        self,
        matrix_size=32,
        patch_size=4,
        in_channels=2,
        embed_dim=128,
        dtype=torch.float64,
    ):
        super().__init__()
        self.matrix_size = matrix_size
        self.patch_size = patch_size
        self.num_patches = (matrix_size // patch_size) ** 2  # 64

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            dtype=dtype,
        )

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, 8, 8)
        x = x.flatten(2).transpose(1, 2)  # (B, 64, embed_dim)
        return x


class PatchUnembed(nn.Module):
    """Convert 64 patch tokens back to 32x32 matrix."""

    def __init__(
        self,
        matrix_size=32,
        patch_size=4,
        out_channels=2,
        embed_dim=128,
        dtype=torch.float64,
    ):
        super().__init__()
        self.matrix_size = matrix_size
        self.patch_size = patch_size
        self.grid_size = matrix_size // patch_size  # 8

        self.proj = nn.Linear(
            embed_dim, patch_size * patch_size * out_channels, dtype=dtype
        )
        self.out_channels = out_channels

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x)  # (B, 64, patch_size^2 * out_channels)

        x = x.view(
            B,
            self.grid_size,
            self.grid_size,
            self.patch_size,
            self.patch_size,
            self.out_channels,
        )
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, self.out_channels, self.matrix_size, self.matrix_size)
        return x


class TransformerBlock(nn.Module):
    """Standard Transformer block with pre-norm."""

    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1, dtype=torch.float64):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, dtype=dtype)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True, dtype=dtype
        )
        self.norm2 = nn.LayerNorm(embed_dim, dtype=dtype)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim, dtype=dtype),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim, dtype=dtype),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class HierarchicalTransformer5Qubit(nn.Module):
    """
    Hierarchical Transformer for 5-qubit density matrix denoising.

    Architecture:
    - Patch embedding: 32x32 -> 64 tokens (4x4 patches)
    - Transformer encoder: 4 layers
    - Bottleneck: embed_dim -> embed_dim/2 -> embed_dim
    - Transformer decoder: 4 layers
    - Patch unembed: 64 tokens -> 32x32

    Parameters: ~1,092,960
    """

    def __init__(
        self,
        loss_fn=None,
        embed_dim=128,
        ffn_dim=256,
        num_heads=8,
        num_layers=4,
        dtype=torch.float64,
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.dtype = dtype

        # Patch embedding
        self.patch_embed = PatchEmbed(
            matrix_size=32,
            patch_size=4,
            in_channels=2,
            embed_dim=embed_dim,
            dtype=dtype,
        )

        # Positional embedding for 64 patches
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, embed_dim, dtype=dtype))

        # Encoder
        self.encoder = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, ffn_dim, dtype=dtype)
                for _ in range(num_layers)
            ]
        )

        # Bottleneck
        self.down = nn.Linear(embed_dim, embed_dim // 2, dtype=dtype)
        self.up = nn.Linear(embed_dim // 2, embed_dim, dtype=dtype)

        # Decoder
        self.decoder = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, ffn_dim, dtype=dtype)
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

        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: (B, 2, 32, 32)
        tokens = self.patch_embed(x)  # (B, 64, embed_dim)
        tokens = tokens + self.pos_embed

        # Encoder
        for block in self.encoder:
            tokens = block(tokens)

        # Bottleneck
        z = self.up(F.gelu(self.down(tokens)))

        # Decoder
        for block in self.decoder:
            z = block(z)

        # Unembed
        out = self.patch_unembed(z)  # (B, 2, 32, 32)
        return out

    def compute_loss(self, pred, target):
        if self.loss_fn is None:
            raise ValueError("loss_fn not set")
        return self.loss_fn(pred, target)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = HierarchicalTransformer5Qubit()  # Already float64 by default
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Model dtype: {next(model.parameters()).dtype}")

    x = torch.randn(4, 2, 32, 32, dtype=torch.float64)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}, Dtype: {y.dtype}")
    assert y.shape == (4, 2, 32, 32)
    assert y.dtype == torch.float64
    print("OK")
