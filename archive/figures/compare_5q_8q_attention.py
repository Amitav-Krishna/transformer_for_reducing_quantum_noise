#!/usr/bin/env python3
"""
Compare attention patterns between 5-qubit and 8-qubit hierarchical transformers.

Goal: Show that the 5-qubit model learned meaningful structure (non-uniform attention)
while the 8-qubit model has essentially random/uniform attention (failed to learn).

This provides visual evidence for why the 8-qubit model underperforms.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Import models
from train_16.models.transformer_5qubit import HierarchicalTransformer5Qubit
from train_16.models.transformer_8qubit import HierarchicalTransformer8Qubit


class AttentionCaptureWrapper5Q(nn.Module):
    """Wrapper that captures attention weights from 5-qubit hierarchical transformer."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.attention_maps = []

    def forward(self, x):
        self.attention_maps = []

        # Patch embedding
        tokens = self.model.patch_embed(x)  # (B, 64, embed_dim)
        tokens = tokens + self.model.pos_embed

        # Encoder - capture attention
        for i, block in enumerate(self.model.encoder):
            x_norm = block.norm1(tokens)
            attn_out, attn_weights = block.attn(
                x_norm, x_norm, x_norm, need_weights=True, average_attn_weights=False
            )
            self.attention_maps.append(
                {
                    "layer": i,
                    "type": "encoder",
                    "weights": attn_weights.detach(),  # (B, heads, 64, 64)
                }
            )
            tokens = tokens + attn_out
            tokens = tokens + block.ffn(block.norm2(tokens))

        # Bottleneck
        z = self.model.up(nn.functional.gelu(self.model.down(tokens)))

        # Decoder - capture attention
        for i, block in enumerate(self.model.decoder):
            x_norm = block.norm1(z)
            attn_out, attn_weights = block.attn(
                x_norm, x_norm, x_norm, need_weights=True, average_attn_weights=False
            )
            self.attention_maps.append(
                {"layer": i, "type": "decoder", "weights": attn_weights.detach()}
            )
            z = z + attn_out
            z = z + block.ffn(block.norm2(z))

        # Unembed
        out = self.model.patch_unembed(z)
        return out


class AttentionCaptureWrapper8Q(nn.Module):
    """Wrapper that captures attention weights from 8-qubit hierarchical transformer."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.attention_maps = []

    def forward(self, x):
        self.attention_maps = []

        # Patch embedding
        tokens = self.model.patch_embed(x)  # (B, 64, embed_dim)
        tokens = tokens + self.model.pos_embed

        # Encoder - capture attention
        for i, block in enumerate(self.model.encoder):
            x_norm = block.norm1(tokens)
            attn_out, attn_weights = block.attn(
                x_norm, x_norm, x_norm, need_weights=True, average_attn_weights=False
            )
            self.attention_maps.append(
                {
                    "layer": i,
                    "type": "encoder",
                    "weights": attn_weights.detach(),  # (B, heads, 64, 64)
                }
            )
            tokens = tokens + attn_out
            tokens = tokens + block.ffn(block.norm2(tokens))

        # Bottleneck
        z = self.model.up(nn.functional.gelu(self.model.down(tokens)))

        # Decoder - capture attention
        for i, block in enumerate(self.model.decoder):
            x_norm = block.norm1(z)
            attn_out, attn_weights = block.attn(
                x_norm, x_norm, x_norm, need_weights=True, average_attn_weights=False
            )
            self.attention_maps.append(
                {"layer": i, "type": "decoder", "weights": attn_weights.detach()}
            )
            z = z + attn_out
            z = z + block.ffn(block.norm2(z))

        # Unembed
        out = self.model.patch_unembed(z)
        return out


def compute_attention_entropy(attn_weights):
    """
    Compute entropy of attention distribution.

    Higher entropy = more uniform attention (less learned structure).
    Lower entropy = more peaked attention (learned specific patterns).

    Max entropy for 64 tokens = log(64) ≈ 4.16
    """
    # attn_weights: (heads, 64, 64)
    # Normalize each row to be a probability distribution
    attn = attn_weights.mean(dim=0)  # Average across heads: (64, 64)

    # Each row is attention from one query to all keys
    # Compute entropy for each row
    attn = attn + 1e-10  # Avoid log(0)
    entropy_per_row = -torch.sum(attn * torch.log(attn), dim=-1)  # (64,)

    return entropy_per_row.mean().item()


def compute_attention_sparsity(attn_weights, top_k=5):
    """
    What fraction of total attention is captured by top-k keys?

    Higher = more sparse/peaked attention (learned structure).
    Lower = more uniform attention.
    """
    attn = attn_weights.mean(dim=0)  # (64, 64)

    # For each query, get top-k attention weights
    topk_vals, _ = torch.topk(attn, k=top_k, dim=-1)  # (64, top_k)
    topk_sum = topk_vals.sum(dim=-1)  # (64,)

    return topk_sum.mean().item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Try to find checkpoints
    checkpoint_5q = None
    checkpoint_8q = None

    # Check remote-synced locations first, then local
    paths_5q = [
        "train_16/checkpoints_16/transformer_5qubit/best.pt",
    ]
    paths_8q = [
        "train_16/checkpoints_16/transformer_8qubit/best.pt",
    ]

    for path in paths_5q:
        if os.path.exists(path):
            checkpoint_5q = path
            break

    for path in paths_8q:
        if os.path.exists(path):
            checkpoint_8q = path
            break

    if checkpoint_5q is None:
        print("ERROR: Cannot find 5-qubit checkpoint locally.")
        print("Please download from remote pod first:")
        print(
            "  scp -P 20634 root@157.157.221.29:/workspace/checkpoints_16/transformer_5qubit/best.pt train_16/checkpoints_16/transformer_5qubit/"
        )
        return

    if checkpoint_8q is None:
        print("ERROR: Cannot find 8-qubit checkpoint locally.")
        print("Please download from remote pod first:")
        print(
            "  scp -P 20657 root@157.157.221.29:/workspace/checkpoints_16/transformer_8qubit/best.pt train_16/checkpoints_16/transformer_8qubit/"
        )
        return

    print(f"Loading 5-qubit model from: {checkpoint_5q}")
    print(f"Loading 8-qubit model from: {checkpoint_8q}")

    # Load models
    model_5q = HierarchicalTransformer5Qubit(dtype=torch.float64)
    model_5q.load_state_dict(
        torch.load(checkpoint_5q, map_location=device, weights_only=False)
    )
    model_5q = model_5q.to(device)
    model_5q.eval()

    model_8q = HierarchicalTransformer8Qubit(dtype=torch.float64)
    model_8q.load_state_dict(
        torch.load(checkpoint_8q, map_location=device, weights_only=False)
    )
    model_8q = model_8q.to(device)
    model_8q.eval()

    # Wrap for attention capture
    wrapper_5q = AttentionCaptureWrapper5Q(model_5q).to(device)
    wrapper_8q = AttentionCaptureWrapper8Q(model_8q).to(device)
    wrapper_5q.eval()
    wrapper_8q.eval()

    # Create test inputs (random density-matrix-like tensors)
    # For fair comparison, use similar normalized inputs
    torch.manual_seed(42)

    # 5-qubit input: (1, 2, 32, 32)
    x_5q = torch.randn(1, 2, 32, 32, dtype=torch.float64, device=device)
    x_5q = x_5q / x_5q.norm() * 0.1  # Normalize

    # 8-qubit input: (1, 2, 256, 256)
    x_8q = torch.randn(1, 2, 256, 256, dtype=torch.float64, device=device)
    x_8q = x_8q / x_8q.norm() * 0.1

    # Run forward pass
    with torch.no_grad():
        _ = wrapper_5q(x_5q)
        _ = wrapper_8q(x_8q)

    print(f"\n5-qubit: Captured {len(wrapper_5q.attention_maps)} attention maps")
    print(f"8-qubit: Captured {len(wrapper_8q.attention_maps)} attention maps")

    # Compute statistics for each layer
    print("\n" + "=" * 70)
    print("ATTENTION ENTROPY COMPARISON")
    print("(Higher entropy = more uniform = less learned structure)")
    print("=" * 70)
    print(
        f"{'Layer':<20} {'5-qubit Entropy':<20} {'8-qubit Entropy':<20} {'Difference':<15}"
    )
    print("-" * 70)

    entropies_5q = []
    entropies_8q = []

    for i, (attn_5q, attn_8q) in enumerate(
        zip(wrapper_5q.attention_maps, wrapper_8q.attention_maps)
    ):
        ent_5q = compute_attention_entropy(attn_5q["weights"][0])
        ent_8q = compute_attention_entropy(attn_8q["weights"][0])
        entropies_5q.append(ent_5q)
        entropies_8q.append(ent_8q)

        layer_name = f"{attn_5q['type']} layer {attn_5q['layer']}"
        diff = ent_8q - ent_5q
        print(f"{layer_name:<20} {ent_5q:<20.4f} {ent_8q:<20.4f} {diff:+.4f}")

    max_entropy = np.log(64)
    print("-" * 70)
    print(
        f"{'Average':<20} {np.mean(entropies_5q):<20.4f} {np.mean(entropies_8q):<20.4f}"
    )
    print(f"{'Max possible':<20} {max_entropy:<20.4f} {max_entropy:<20.4f}")
    print(f"{'% of max (5q)':<20} {100 * np.mean(entropies_5q) / max_entropy:.1f}%")
    print(f"{'% of max (8q)':<20} {100 * np.mean(entropies_8q) / max_entropy:.1f}%")

    # Compute sparsity (top-5 attention concentration)
    print("\n" + "=" * 70)
    print("ATTENTION SPARSITY (Top-5 Concentration)")
    print("(Higher = more peaked attention = learned specific patterns)")
    print("=" * 70)
    print(f"{'Layer':<20} {'5-qubit Top-5':<20} {'8-qubit Top-5':<20}")
    print("-" * 70)

    for i, (attn_5q, attn_8q) in enumerate(
        zip(wrapper_5q.attention_maps, wrapper_8q.attention_maps)
    ):
        spar_5q = compute_attention_sparsity(attn_5q["weights"][0])
        spar_8q = compute_attention_sparsity(attn_8q["weights"][0])

        layer_name = f"{attn_5q['type']} layer {attn_5q['layer']}"
        print(f"{layer_name:<20} {spar_5q:<20.4f} {spar_8q:<20.4f}")

    # Create visualization
    os.makedirs("figures", exist_ok=True)

    # Plot side-by-side attention maps for encoder layer 0
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # Get encoder layer 0 attention, averaged across heads
    attn_5q_enc0 = wrapper_5q.attention_maps[0]["weights"][0].mean(dim=0).cpu().numpy()
    attn_8q_enc0 = wrapper_8q.attention_maps[0]["weights"][0].mean(dim=0).cpu().numpy()

    # 5-qubit
    ax = axes[0]
    im = ax.imshow(attn_5q_enc0, cmap="viridis", aspect="equal")
    ax.set_title("5-Qubit Transformer\n(Encoder Layer 0)", fontsize=14)
    ax.set_xlabel("Key (Patch Index)", fontsize=12)
    ax.set_ylabel("Query (Patch Index)", fontsize=12)
    fig.colorbar(im, ax=ax, label="Attention Weight")

    # Add entropy annotation
    ent_5q = compute_attention_entropy(wrapper_5q.attention_maps[0]["weights"][0])
    ax.text(
        0.02,
        0.98,
        f"Entropy: {ent_5q:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # 8-qubit
    ax = axes[1]
    im = ax.imshow(attn_8q_enc0, cmap="viridis", aspect="equal")
    ax.set_title("8-Qubit Transformer\n(Encoder Layer 0)", fontsize=14)
    ax.set_xlabel("Key (Patch Index)", fontsize=12)
    ax.set_ylabel("Query (Patch Index)", fontsize=12)
    fig.colorbar(im, ax=ax, label="Attention Weight")

    ent_8q = compute_attention_entropy(wrapper_8q.attention_maps[0]["weights"][0])
    ax.text(
        0.02,
        0.98,
        f"Entropy: {ent_8q:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.suptitle(
        "Attention Pattern Comparison: 5-Qubit vs 8-Qubit Hierarchical Transformers",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("figures/5q_vs_8q_attention_comparison.pdf")
    plt.savefig("figures/5q_vs_8q_attention_comparison.png")
    print("\nSaved: figures/5q_vs_8q_attention_comparison.pdf")

    # Plot attention distribution histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    ax = axes[0]
    ax.hist(attn_5q_enc0.flatten(), bins=50, alpha=0.7, color="blue", label="5-qubit")
    ax.set_xlabel("Attention Weight", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("5-Qubit Attention Distribution", fontsize=14)
    ax.axvline(x=1 / 64, color="red", linestyle="--", label=f"Uniform = {1 / 64:.4f}")
    ax.legend()

    ax = axes[1]
    ax.hist(attn_8q_enc0.flatten(), bins=50, alpha=0.7, color="orange", label="8-qubit")
    ax.set_xlabel("Attention Weight", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("8-Qubit Attention Distribution", fontsize=14)
    ax.axvline(x=1 / 64, color="red", linestyle="--", label=f"Uniform = {1 / 64:.4f}")
    ax.legend()

    plt.suptitle("Attention Weight Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/5q_vs_8q_attention_histogram.pdf")
    plt.savefig("figures/5q_vs_8q_attention_histogram.png")
    print("Saved: figures/5q_vs_8q_attention_histogram.pdf")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"5-qubit transformer: Mean entropy = {np.mean(entropies_5q):.3f} ({100 * np.mean(entropies_5q) / max_entropy:.1f}% of max)"
    )
    print(
        f"8-qubit transformer: Mean entropy = {np.mean(entropies_8q):.3f} ({100 * np.mean(entropies_8q) / max_entropy:.1f}% of max)"
    )

    if np.mean(entropies_8q) > np.mean(entropies_5q):
        diff_pct = (
            100
            * (np.mean(entropies_8q) - np.mean(entropies_5q))
            / np.mean(entropies_5q)
        )
        print(f"\n8-qubit attention is {diff_pct:.1f}% MORE uniform (higher entropy)")
        print(
            "This suggests the 8-qubit model failed to learn meaningful patch correlations."
        )
    else:
        print("\nUnexpected: 8-qubit has lower entropy than 5-qubit")


if __name__ == "__main__":
    main()
