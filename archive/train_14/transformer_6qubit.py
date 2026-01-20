"""
Hierarchical Transformer for 6-qubit density matrices.

Addresses the scalability critique: element-wise tokenization (4096 tokens for 6 qubits)
is too expensive. We use hierarchical tokenization instead:

1. Patch embedding: 64×64 matrix → 8×8 patches → 64 tokens
2. Standard attention over 64 tokens (feasible: O(64²) = 4096 ops)
3. Upsample back to 64×64

This demonstrates that the Transformer approach can scale beyond 5 qubits
with appropriate tokenization strategies.

For comparison:
- Element-wise (5 qubits): 1024 tokens, O(1M) attention
- Element-wise (6 qubits): 4096 tokens, O(16M) attention (expensive)
- Hierarchical (6 qubits): 64 tokens, O(4k) attention (feasible)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """Convert 64×64 matrix to 64 patch tokens (8×8 patches)."""

    def __init__(self, matrix_size=64, patch_size=8, in_channels=2, embed_dim=128):
        super().__init__()
        self.matrix_size = matrix_size
        self.patch_size = patch_size
        self.num_patches = (matrix_size // patch_size) ** 2  # 64

        # Embed each 8×8×2 patch into embed_dim
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, 2, 64, 64)
        x = self.proj(x)  # (B, embed_dim, 8, 8)
        x = x.flatten(2).transpose(1, 2)  # (B, 64, embed_dim)
        return x


class PatchUnembed(nn.Module):
    """Convert 64 patch tokens back to 64×64 matrix."""

    def __init__(self, matrix_size=64, patch_size=8, out_channels=2, embed_dim=128):
        super().__init__()
        self.matrix_size = matrix_size
        self.patch_size = patch_size
        self.grid_size = matrix_size // patch_size  # 8

        # Project back to patch size
        self.proj = nn.Linear(embed_dim, patch_size * patch_size * out_channels)
        self.out_channels = out_channels

    def forward(self, x):
        # x: (B, 64, embed_dim)
        B = x.shape[0]
        x = self.proj(x)  # (B, 64, patch_size² * out_channels)

        # Reshape to spatial
        x = x.view(B, self.grid_size, self.grid_size,
                   self.patch_size, self.patch_size, self.out_channels)
        # Rearrange to (B, out_channels, H, W)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, self.out_channels, self.matrix_size, self.matrix_size)
        return x


class TransformerBlock(nn.Module):
    """Standard Transformer block with pre-norm."""

    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads,
            dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


class HierarchicalTransformer6Qubit(nn.Module):
    """
    Hierarchical Transformer for 6-qubit density matrix denoising.

    Architecture:
    - Patch embedding: 64×64 → 64 tokens (8×8 patches)
    - Transformer encoder-decoder: 4 layers each
    - Bottleneck: embed_dim → embed_dim/2 → embed_dim
    - Patch unembed: 64 tokens → 64×64

    Parameters: ~500k (comparable to 5-qubit models)
    """

    def __init__(self, loss_fn=None, embed_dim=128, ffn_dim=256, num_heads=8, num_layers=4):
        super().__init__()
        self.loss_fn = loss_fn

        # Patch embedding
        self.patch_embed = PatchEmbed(
            matrix_size=64, patch_size=8,
            in_channels=2, embed_dim=embed_dim
        )

        # Positional embedding for 64 patches
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, embed_dim))

        # Encoder
        self.encoder = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])

        # Bottleneck
        self.down = nn.Linear(embed_dim, embed_dim // 2)
        self.up = nn.Linear(embed_dim // 2, embed_dim)

        # Decoder
        self.decoder = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])

        # Output projection
        self.patch_unembed = PatchUnembed(
            matrix_size=64, patch_size=8,
            out_channels=2, embed_dim=embed_dim
        )

        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: (B, 2, 64, 64)
        B = x.shape[0]

        # Patch embed
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
        out = self.patch_unembed(z)  # (B, 2, 64, 64)

        return out

    def compute_loss(self, pred, target):
        return self.loss_fn(pred, target)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from losses.frob import FrobeniusFidelityLoss

    model = HierarchicalTransformer6Qubit(loss_fn=FrobeniusFidelityLoss())
    print(f"HierarchicalTransformer6Qubit parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(4, 2, 64, 64)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")

    # Verify shapes
    assert y.shape == (4, 2, 64, 64), f"Expected (4, 2, 64, 64), got {y.shape}"
    print("Forward pass successful!")
