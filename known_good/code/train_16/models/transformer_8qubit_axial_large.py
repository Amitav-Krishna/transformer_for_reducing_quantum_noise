"""
Large Axial Attention Transformer for 8-qubit density matrices (~114M params).

Scaled up from the base model:
- embed_dim: 128 → 768
- ffn_dim: 256 → 3072
- num_heads: 8 → 12
- num_layers: 4 → 6 (encoder) + 6 (decoder)

This keeps the same 1024 tokens (8×8 patches on 256×256), so attention
cost stays manageable: O(2 × 32³) = 65k ops per layer.

The extra params come from wider embeddings and FFN, which parallelize well.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbed8x8(nn.Module):
    """Convert 256×256 matrix to 1024 patch tokens (8×8 patches)."""

    def __init__(
        self,
        matrix_size=256,
        patch_size=8,
        in_channels=2,
        embed_dim=768,
    ):
        super().__init__()
        self.matrix_size = matrix_size
        self.patch_size = patch_size
        self.grid_size = matrix_size // patch_size  # 32
        self.num_patches = self.grid_size**2  # 1024

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        # x: (B, 2, 256, 256)
        x = self.proj(x)  # (B, embed_dim, 32, 32)
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).transpose(1, 2)  # (B, 1024, embed_dim)
        return x


class PatchUnembed8x8(nn.Module):
    """Convert 1024 patch tokens back to 256×256 matrix."""

    def __init__(
        self,
        matrix_size=256,
        patch_size=8,
        out_channels=2,
        embed_dim=768,
    ):
        super().__init__()
        self.matrix_size = matrix_size
        self.patch_size = patch_size
        self.grid_size = matrix_size // patch_size  # 32
        self.out_channels = out_channels

        self.proj = nn.Linear(embed_dim, patch_size * patch_size * out_channels)

    def forward(self, x):
        # x: (B, 1024, embed_dim)
        B = x.shape[0]
        x = self.proj(x)  # (B, 1024, 8*8*2)

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


class AxialAttention(nn.Module):
    """
    Axial attention: factorized attention along rows then columns.

    For a 32×32 grid of tokens:
    - Row attention: each of 32 rows does attention over 32 tokens
    - Column attention: each of 32 columns does attention over 32 tokens

    Total: 2 × 32 × 32² = 65,536 attention ops (vs 1,048,576 for dense)
    """

    def __init__(self, embed_dim, num_heads, grid_size=32, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.grid_size = grid_size
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        # Separate projections for row and column attention
        self.row_qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.row_proj = nn.Linear(embed_dim, embed_dim)

        self.col_qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.col_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, 1024, embed_dim) = (B, 32*32, embed_dim)
        B, N, C = x.shape
        G = self.grid_size  # 32

        # Reshape to grid: (B, 32, 32, embed_dim)
        x_grid = x.view(B, G, G, C)

        # === Row attention ===
        x_rows = x_grid.view(B * G, G, C)

        qkv = self.row_qkv(x_rows).reshape(B * G, G, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*G, heads, G, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        row_out = (attn @ v).transpose(1, 2).reshape(B * G, G, C)
        row_out = self.row_proj(row_out)
        row_out = row_out.view(B, G, G, C)

        # === Column attention ===
        x_cols = x_grid.permute(0, 2, 1, 3).contiguous().view(B * G, G, C)

        qkv = self.col_qkv(x_cols).reshape(B * G, G, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        col_out = (attn @ v).transpose(1, 2).reshape(B * G, G, C)
        col_out = self.col_proj(col_out)
        col_out = col_out.view(B, G, G, C).permute(0, 2, 1, 3).contiguous()

        # Combine row and column attention outputs
        out = row_out + col_out

        # Flatten back: (B, 32, 32, C) -> (B, 1024, C)
        return out.view(B, N, C)


class AxialTransformerBlock(nn.Module):
    """Transformer block with axial attention + FFN."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        ffn_dim,
        grid_size=32,
        dropout=0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = AxialAttention(embed_dim, num_heads, grid_size, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class AxialTransformer8QubitLarge(nn.Module):
    """
    Large Axial Attention Transformer for 8-qubit density matrix denoising.

    ~114M parameters with:
    - embed_dim=768
    - ffn_dim=3072
    - num_heads=12
    - num_layers=6 (encoder) + 6 (decoder) = 12 total

    Keeps same 1024 tokens as base model, so attention cost is unchanged.
    """

    def __init__(
        self,
        loss_fn=None,
        embed_dim=768,
        ffn_dim=3072,
        num_heads=12,
        num_layers=6,  # per encoder/decoder
        dropout=0.1,
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.grid_size = 32  # 256 / 8

        # Patch embedding: 8×8 patches
        self.patch_embed = PatchEmbed8x8(
            matrix_size=256,
            patch_size=8,
            in_channels=2,
            embed_dim=embed_dim,
        )

        # 2D positional embedding for 32×32 grid
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.grid_size * self.grid_size, embed_dim)
        )

        # Encoder with axial attention
        self.encoder = nn.ModuleList(
            [
                AxialTransformerBlock(
                    embed_dim, num_heads, ffn_dim, self.grid_size, dropout
                )
                for _ in range(num_layers)
            ]
        )

        # Decoder with axial attention
        self.decoder = nn.ModuleList(
            [
                AxialTransformerBlock(
                    embed_dim, num_heads, ffn_dim, self.grid_size, dropout
                )
                for _ in range(num_layers)
            ]
        )

        # Output norm + unembed
        self.norm = nn.LayerNorm(embed_dim)
        self.patch_unembed = PatchUnembed8x8(
            matrix_size=256,
            patch_size=8,
            out_channels=2,
            embed_dim=embed_dim,
        )

        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, 2, 256, 256)
        residual = x

        tokens = self.patch_embed(x)  # (B, 1024, embed_dim)
        tokens = tokens + self.pos_embed

        # Encoder
        for block in self.encoder:
            tokens = block(tokens)

        # Decoder
        for block in self.decoder:
            tokens = block(tokens)

        # Output
        tokens = self.norm(tokens)
        out = self.patch_unembed(tokens)  # (B, 2, 256, 256)

        # Global residual connection
        return residual + out

    def compute_loss(self, pred, target):
        if self.loss_fn is None:
            raise ValueError("loss_fn not set")
        return self.loss_fn(pred, target)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = AxialTransformer8QubitLarge()
    print(f"Parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(2, 2, 256, 256)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    assert y.shape == (2, 2, 256, 256)
    print("OK")
