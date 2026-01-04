"""
Axial Attention Transformer v2 for 8-qubit density matrices.

Changes from v1:
1. Residual prediction (output = input + model(input))
2. Row/col attention merged via concat + linear (not sum)
3. Dropout 0.1 throughout
"""

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(2, embed_dim, kernel_size=8, stride=8)

    def forward(self, x):
        # x: (B, 2, 256, 256) -> (B, embed_dim, 32, 32) -> (B, 1024, embed_dim)
        return self.proj(x).flatten(2).transpose(1, 2)


class PatchUnembed(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.proj = nn.Linear(embed_dim, 8 * 8 * 2)

    def forward(self, x):
        # x: (B, 1024, embed_dim) -> (B, 2, 256, 256)
        B = x.shape[0]
        x = self.proj(x)  # (B, 1024, 128)
        x = x.view(B, 32, 32, 8, 8, 2)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(B, 2, 256, 256)


class AxialAttention(nn.Module):
    """
    Axial attention with concat+linear merge for row/col outputs.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        # Row attention
        self.row_qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.row_proj = nn.Linear(embed_dim, embed_dim)

        # Column attention
        self.col_qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.col_proj = nn.Linear(embed_dim, embed_dim)

        # Merge row and col via concat + linear
        self.merge = nn.Linear(2 * embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape  # (B, 1024, embed_dim)
        G = 32  # grid size

        x_grid = x.view(B, G, G, C)

        # Row attention: each row attends to itself
        x_rows = x_grid.view(B * G, G, C)
        qkv = self.row_qkv(x_rows).reshape(B * G, G, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*G, heads, G, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        row_out = (attn @ v).transpose(1, 2).reshape(B * G, G, C)
        row_out = self.row_proj(row_out)
        row_out = row_out.view(B, G, G, C)

        # Column attention: each column attends to itself
        x_cols = x_grid.permute(0, 2, 1, 3).contiguous().view(B * G, G, C)
        qkv = self.col_qkv(x_cols).reshape(B * G, G, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        col_out = (attn @ v).transpose(1, 2).reshape(B * G, G, C)
        col_out = self.col_proj(col_out)
        col_out = col_out.view(B, G, G, C).permute(0, 2, 1, 3).contiguous()

        # Merge via concat + linear (not sum)
        merged = torch.cat([row_out, col_out], dim=-1)  # (B, G, G, 2*C)
        merged = self.merge(merged)  # (B, G, G, C)
        merged = self.proj_drop(merged)

        return merged.view(B, N, C)


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1, init_scale=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = AxialAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # LayerScale
        self.gamma1 = nn.Parameter(init_scale * torch.ones(embed_dim))
        self.gamma2 = nn.Parameter(init_scale * torch.ones(embed_dim))

    def forward(self, x):
        x = x + self.gamma1 * self.attn(self.norm1(x))
        x = x + self.gamma2 * self.ffn(self.norm2(x))
        return x


class AxialTransformerV2(nn.Module):
    """
    Axial Transformer v2 for 8-qubit density matrix denoising.

    Key features:
    - Residual prediction: output = input + correction
    - Axial attention with concat+linear merge
    - LayerScale for stable training
    - Dropout throughout
    """

    def __init__(
        self,
        embed_dim=256,
        ffn_dim=1024,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
        init_scale=0.1,
        loss_fn=None,
    ):
        super().__init__()
        self.loss_fn = loss_fn

        self.patch_embed = PatchEmbed(embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                Block(embed_dim, num_heads, ffn_dim, dropout, init_scale)
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.patch_unembed = PatchUnembed(embed_dim)

        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self._init_weights()

    def _init_weights(self):
        # Zero-init output projection for clean residual at start
        nn.init.zeros_(self.patch_unembed.proj.weight)
        nn.init.zeros_(self.patch_unembed.proj.bias)

    def forward(self, x):
        # x: (B, 2, 256, 256)
        residual = x

        tokens = self.patch_embed(x)  # (B, 1024, embed_dim)
        tokens = tokens + self.pos_embed
        tokens = self.pos_drop(tokens)

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)
        correction = self.patch_unembed(tokens)  # (B, 2, 256, 256)

        return residual + correction  # Residual prediction

    def compute_loss(self, pred, target):
        if self.loss_fn is None:
            raise ValueError("loss_fn not set")
        return self.loss_fn(pred, target)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = AxialTransformerV2()
    print(f"Parameters: {count_parameters(model):,}")

    x = torch.randn(2, 2, 256, 256)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    assert y.shape == x.shape
    print("OK")
