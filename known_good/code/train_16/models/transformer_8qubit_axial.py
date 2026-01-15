"""
Axial Attention Transformer for 8-qubit density matrices.

Key design choices (per ChatGPT analysis):
- 8×8 patches on 256×256 matrix → 32×32 = 1024 patch tokens
- Axial attention: row-then-column factorization
- Each element can reach any other in 2 hops (diameter 2 graph)
- Preserves global information flow along physically meaningful axes
  (rows = fixed bra, columns = fixed ket)

Attention cost comparison:
- Dense on 1024 tokens: 1,048,576 ops/layer
- Axial on 32×32 grid: 2 × 32³ = 65,536 ops/layer (16× cheaper)

Architecture:
- Patch embedding: 256×256 → 1024 tokens (8×8 patches, 128 values each)
- Axial encoder: row attention + column attention per layer
- Bottleneck
- Axial decoder
- Patch unembed: 1024 tokens → 256×256
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
        embed_dim=128,
        dtype=torch.float64,
    ):
        super().__init__()
        self.matrix_size = matrix_size
        self.patch_size = patch_size
        self.grid_size = matrix_size // patch_size  # 32
        self.num_patches = self.grid_size**2  # 1024

        # Conv2d with kernel=stride=patch_size extracts non-overlapping patches
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            dtype=dtype,
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
        embed_dim=128,
        dtype=torch.float64,
    ):
        super().__init__()
        self.matrix_size = matrix_size
        self.patch_size = patch_size
        self.grid_size = matrix_size // patch_size  # 32
        self.out_channels = out_channels

        # Project each token to patch_size² × out_channels
        self.proj = nn.Linear(
            embed_dim, patch_size * patch_size * out_channels, dtype=dtype
        )

    def forward(self, x):
        # x: (B, 1024, embed_dim)
        B = x.shape[0]
        x = self.proj(x)  # (B, 1024, 8*8*2)

        # Reshape to (B, 32, 32, 8, 8, 2)
        x = x.view(
            B,
            self.grid_size,
            self.grid_size,
            self.patch_size,
            self.patch_size,
            self.out_channels,
        )
        # Permute to (B, 2, 32, 8, 32, 8) then reshape
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

    def __init__(
        self, embed_dim, num_heads, grid_size=32, dropout=0.1, dtype=torch.float64
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.grid_size = grid_size
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        # Separate projections for row and column attention
        self.row_qkv = nn.Linear(embed_dim, 3 * embed_dim, dtype=dtype)
        self.row_proj = nn.Linear(embed_dim, embed_dim, dtype=dtype)

        self.col_qkv = nn.Linear(embed_dim, 3 * embed_dim, dtype=dtype)
        self.col_proj = nn.Linear(embed_dim, embed_dim, dtype=dtype)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, 1024, embed_dim) = (B, 32*32, embed_dim)
        B, N, C = x.shape
        G = self.grid_size  # 32

        # Reshape to grid: (B, 32, 32, embed_dim)
        x_grid = x.view(B, G, G, C)

        # === Row attention ===
        # Each row of 32 tokens attends to itself
        # Reshape: (B, 32, 32, C) -> (B*32, 32, C)
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

        # Accumulate row attention output
        out = row_out

        # === Column attention ===
        # Each column of 32 tokens attends to itself
        # Transpose grid: (B, 32, 32, C) -> (B, 32, 32, C) with cols as sequence
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

        # Accumulate column attention output (no internal residual - outer block handles it)
        out = out + col_out

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
        dtype=torch.float64,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, dtype=dtype)
        self.attn = AxialAttention(embed_dim, num_heads, grid_size, dropout, dtype)
        self.norm2 = nn.LayerNorm(embed_dim, dtype=dtype)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim, dtype=dtype),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim, dtype=dtype),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pre-norm with residual (axial attention also has internal residuals for row/col)
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class AxialTransformer8Qubit(nn.Module):
    """
    Axial Attention Transformer for 8-qubit density matrix denoising.

    Architecture:
    - Patch embedding: 256×256 → 1024 tokens (8×8 patches)
    - Positional embedding (2D learnable)
    - Axial encoder: 4 layers of row+column attention
    - Bottleneck: compress then expand
    - Axial decoder: 4 layers
    - Patch unembed: 1024 tokens → 256×256

    Key advantages over 32×32 patch hierarchical:
    - 16× more tokens = 16× more spatial resolution
    - Each patch has 128 values (vs 2048) = much less compression
    - Axial attention keeps cost manageable: O(2×32³) vs O(1024²)
    - Global information flow preserved (diameter-2 connectivity)
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
        self.grid_size = 32  # 256 / 8

        # Patch embedding: 8×8 patches
        self.patch_embed = PatchEmbed8x8(
            matrix_size=256,
            patch_size=8,
            in_channels=2,
            embed_dim=embed_dim,
            dtype=dtype,
        )

        # 2D positional embedding for 32×32 grid
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.grid_size * self.grid_size, embed_dim, dtype=dtype)
        )

        # Encoder with axial attention
        self.encoder = nn.ModuleList(
            [
                AxialTransformerBlock(
                    embed_dim, num_heads, ffn_dim, self.grid_size, dtype=dtype
                )
                for _ in range(num_layers)
            ]
        )

        # Bottleneck
        self.down = nn.Linear(embed_dim, embed_dim // 2, dtype=dtype)
        self.up = nn.Linear(embed_dim // 2, embed_dim, dtype=dtype)

        # Decoder with axial attention
        self.decoder = nn.ModuleList(
            [
                AxialTransformerBlock(
                    embed_dim, num_heads, ffn_dim, self.grid_size, dtype=dtype
                )
                for _ in range(num_layers)
            ]
        )

        # Output
        self.patch_unembed = PatchUnembed8x8(
            matrix_size=256,
            patch_size=8,
            out_channels=2,
            embed_dim=embed_dim,
            dtype=dtype,
        )

        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: (B, 2, 256, 256)
        tokens = self.patch_embed(x)  # (B, 1024, embed_dim)
        tokens = tokens + self.pos_embed

        # Encoder
        for block in self.encoder:
            tokens = block(tokens)

        # Bottleneck
        z = self.up(F.gelu(self.down(tokens)))

        # Decoder
        for block in self.decoder:
            z = block(z)

        # Output
        out = self.patch_unembed(z)  # (B, 2, 256, 256)
        return out

    def compute_loss(self, pred, target):
        if self.loss_fn is None:
            raise ValueError("loss_fn not set")
        return self.loss_fn(pred, target)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = AxialTransformer8Qubit()
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Model dtype: {next(model.parameters()).dtype}")

    x = torch.randn(2, 2, 256, 256, dtype=torch.float64)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}, Dtype: {y.dtype}")
    assert y.shape == (2, 2, 256, 256)
    assert y.dtype == torch.float64
    print("OK")
