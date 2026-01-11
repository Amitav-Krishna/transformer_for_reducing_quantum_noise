"""
Hierarchical MLP for 8-qubit density matrices (float64).

Control experiment for scalability: Same architecture as 5-qubit MLP
but with larger patches (32x32 instead of 4x4).

Specs:
- Matrix size: 256x256
- Patch size: 32x32
- Tokens: 64 (same as 5-qubit!)
- Params: ~1,611,072 (matched to 8-qubit Transformer)
- Layers: 4 encoder + 4 decoder
- Token hidden: 384
- Channel hidden: 320
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_16.models.transformer_8qubit import PatchEmbed, PatchUnembed
from train_16.models.mlp_5qubit import MLPMixerBlock


class HierarchicalMLP8Qubit(nn.Module):
    """
    Hierarchical MLP for 8-qubit density matrix denoising.

    Architecture:
    - Patch embedding: 256x256 -> 64 tokens (32x32 patches)
    - MLP-Mixer encoder: 4 layers
    - Bottleneck: embed_dim -> embed_dim/2 -> embed_dim
    - MLP-Mixer decoder: 4 layers
    - Patch unembed: 64 tokens -> 256x256

    Parameters: ~1,611,072 (matched to 8-qubit Transformer)

    Key insight: Same token count (64) as 5-qubit model, demonstrating
    constant-complexity scaling.
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

        # Patch embedding (256x256 -> 64 tokens with 32x32 patches)
        self.patch_embed = PatchEmbed(
            matrix_size=256,
            patch_size=32,
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

        # Output projection (64 tokens -> 256x256)
        self.patch_unembed = PatchUnembed(
            matrix_size=256,
            patch_size=32,
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
    model = HierarchicalMLP8Qubit()  # Already float64 by default
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Model dtype: {next(model.parameters()).dtype}")

    x = torch.randn(4, 2, 256, 256, dtype=torch.float64)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}, Dtype: {y.dtype}")
    assert y.shape == (4, 2, 256, 256)
    assert y.dtype == torch.float64
    print("OK")
