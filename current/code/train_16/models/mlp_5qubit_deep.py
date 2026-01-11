"""
Deep Hierarchical MLP for 5-qubit density matrices (float64).

Depth control experiment: 2x more layers than Transformer (8+8 vs 4+4).
If this still underperforms Transformer, depth is not the issue.

Specs:
- Matrix size: 32x32
- Patch size: 4x4
- Tokens: 64
- Params: ~2,151,776 (2x matched)
- Layers: 8 encoder + 8 decoder (2x depth)
- Token hidden: 384 (same as matched)
- Channel hidden: 320 (same as matched)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_16.models.transformer_5qubit import PatchEmbed, PatchUnembed
from train_16.models.mlp_5qubit import MLPMixerBlock


class HierarchicalMLP5QubitDeep(nn.Module):
    """
    Deep Hierarchical MLP for 5-qubit density matrix denoising.

    Architecture:
    - Patch embedding: 32x32 -> 64 tokens (4x4 patches)
    - MLP-Mixer encoder: 8 layers (2x Transformer depth)
    - Bottleneck: embed_dim -> embed_dim/2 -> embed_dim
    - MLP-Mixer decoder: 8 layers (2x Transformer depth)
    - Patch unembed: 64 tokens -> 32x32

    Parameters: ~2,151,776 (2x matched MLP/Transformer)

    Key test: If MLP fails because it needs more "processing steps",
    doubling the depth should help. If it still loses, the advantage
    is truly about attention's input-dependent mixing.
    """

    def __init__(
        self,
        loss_fn=None,
        embed_dim=128,
        token_hidden_dim=384,  # Same as matched
        channel_hidden_dim=320,  # Same as matched
        num_layers=8,  # 2x Transformer depth
        dtype=torch.float64,
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.dtype = dtype
        self.num_patches = 64

        # Patch embedding
        self.patch_embed = PatchEmbed(
            matrix_size=32,
            patch_size=4,
            in_channels=2,
            embed_dim=embed_dim,
            dtype=dtype,
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, embed_dim, dtype=dtype))

        # Encoder (8 layers)
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

        # Decoder (8 layers)
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
    model = HierarchicalMLP5QubitDeep()  # Already float64 by default
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Model dtype: {next(model.parameters()).dtype}")

    x = torch.randn(4, 2, 32, 32, dtype=torch.float64)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}, Dtype: {y.dtype}")
    assert y.shape == (4, 2, 32, 32)
    assert y.dtype == torch.float64
    print("OK")
