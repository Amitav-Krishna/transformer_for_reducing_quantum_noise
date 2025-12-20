#!/usr/bin/env python3
"""
Visualize attention maps from the trained Transformer on entangled states.

Shows which density matrix elements attend to which other elements,
demonstrating whether the model learns entanglement structure.

Usage:
    python figures/attention_maps.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Will be populated by hooks
attention_weights = {}


def get_attention_hook(name):
    """Create a hook that captures attention weights."""
    def hook(module, input, output):
        # For nn.MultiheadAttention, output is (attn_output, attn_weights)
        # But we need to call it with need_weights=True
        # Instead, we'll monkey-patch the forward pass
        pass
    return hook


class AttentionCaptureWrapper(nn.Module):
    """Wrapper that captures attention weights from TransformerEncoder."""

    def __init__(self, transformer_model):
        super().__init__()
        self.model = transformer_model
        self.attention_maps = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on attention layers to capture weights."""
        self._hooks = []
        self._attn_outputs = {}

        def make_hook(name):
            def hook(module, input, output):
                # MultiheadAttention returns (attn_output, attn_weights) when called directly
                # But when used in TransformerEncoderLayer, weights aren't returned by default
                # So we need to call forward manually with need_weights=True
                pass
            return hook

        # We'll capture attention in forward() instead

    def forward(self, x):
        """Forward pass that captures attention weights."""
        B = x.shape[0]
        # (B, 2, 32, 32) -> (B, 1024, 2)
        x = x.permute(0, 2, 3, 1).reshape(B, 1024, 2)

        x = self.model.input_proj(x) + self.model.pos_embedding

        # Manually run through encoder layers to capture attention
        self.attention_maps = []
        enc = x
        for i, layer in enumerate(self.model.encoder.layers):
            # Get attention weights from self-attention
            attn_out, attn_weights = layer.self_attn(
                enc, enc, enc,
                need_weights=True,
                average_attn_weights=False
            )
            self.attention_maps.append({
                'layer': i,
                'type': 'encoder_self_attn',
                'weights': attn_weights.detach()  # (B, heads, seq, seq)
            })

            # Complete the layer forward pass (matching TransformerEncoderLayer implementation)
            # Post-norm architecture: x = norm(x + sublayer(x))
            x2 = layer.norm1(enc + layer.dropout1(attn_out))

            # Feed-forward block
            ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(x2))))
            enc = layer.norm2(x2 + layer.dropout2(ff_out))

        z = self.model.up(self.model.down(enc))

        # Decoder with cross-attention
        dec = z
        for i, layer in enumerate(self.model.decoder.layers):
            # Self-attention in decoder
            self_attn_out, self_attn_weights = layer.self_attn(
                dec, dec, dec,
                need_weights=True,
                average_attn_weights=False
            )
            self.attention_maps.append({
                'layer': i,
                'type': 'decoder_self_attn',
                'weights': self_attn_weights.detach()
            })
            x2 = layer.norm1(dec + layer.dropout1(self_attn_out))

            # Cross-attention to encoder output
            cross_attn_out, cross_attn_weights = layer.multihead_attn(
                x2, enc, enc,
                need_weights=True,
                average_attn_weights=False
            )
            self.attention_maps.append({
                'layer': i,
                'type': 'decoder_cross_attn',
                'weights': cross_attn_weights.detach()
            })
            x3 = layer.norm2(x2 + layer.dropout2(cross_attn_out))

            # Feed-forward block
            ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(x3))))
            dec = layer.norm3(x3 + layer.dropout3(ff_out))

        out = self.model.output_proj(dec)
        out = out.reshape(B, 32, 32, 2)
        out = out.permute(0, 3, 1, 2)
        return out


def create_ghz_state(n_qubits=5):
    """
    Create a GHZ state density matrix: |GHZ⟩ = (|00...0⟩ + |11...1⟩) / √2

    Returns: (2, 32, 32) tensor (real, imag channels)
    """
    dim = 2 ** n_qubits  # 32 for 5 qubits
    rho = torch.zeros(dim, dim, dtype=torch.complex128)

    # |00000⟩ = index 0, |11111⟩ = index 31
    # |GHZ⟩⟨GHZ| = 0.5 * (|0⟩⟨0| + |0⟩⟨1| + |1⟩⟨0| + |1⟩⟨1|)
    # where |0⟩ = |00000⟩ and |1⟩ = |11111⟩
    rho[0, 0] = 0.5      # |00000⟩⟨00000|
    rho[31, 31] = 0.5    # |11111⟩⟨11111|
    rho[0, 31] = 0.5     # |00000⟩⟨11111| (coherence)
    rho[31, 0] = 0.5     # |11111⟩⟨00000| (coherence)

    # Convert to 2-channel format
    result = torch.zeros(2, dim, dim, dtype=torch.float32)
    result[0] = rho.real.float()
    result[1] = rho.imag.float()

    return result


def create_bell_like_state():
    """
    Create a state with Bell-like correlations between specific qubit pairs.
    For 5 qubits, create entanglement between qubits 0-4 and 1-3.

    Returns: (2, 32, 32) tensor
    """
    dim = 32
    rho = torch.zeros(dim, dim, dtype=torch.complex128)

    # Create a more complex entangled state
    # Superposition of computational basis states with entanglement
    states = [0b00000, 0b10001, 0b01010, 0b11011]  # Correlated patterns
    n_states = len(states)

    for i in states:
        for j in states:
            rho[i, j] = 1.0 / n_states

    # Normalize trace
    rho = rho / torch.trace(rho)

    result = torch.zeros(2, dim, dim, dtype=torch.float32)
    result[0] = rho.real.float()
    result[1] = rho.imag.float()

    return result


def add_depolarizing_noise(rho, p=0.1):
    """Add depolarizing noise to a density matrix."""
    dim = rho.shape[1]
    # Convert to complex
    rho_complex = rho[0] + 1j * rho[1]

    # Depolarizing channel: (1-p)*rho + p*I/d
    identity = torch.eye(dim, dtype=torch.complex128) / dim
    rho_noisy = (1 - p) * rho_complex + p * identity

    result = torch.zeros_like(rho)
    result[0] = rho_noisy.real.float()
    result[1] = rho_noisy.imag.float()

    return result


def plot_attention_map(attn_weights, title, save_path, highlight_indices=None):
    """
    Plot a single attention map.

    Args:
        attn_weights: (seq_len, seq_len) attention matrix
        title: Plot title
        save_path: Where to save
        highlight_indices: List of indices to highlight (e.g., [0, 31] for GHZ)
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    # Use log scale for better visibility of patterns
    attn_np = attn_weights.cpu().numpy()
    attn_np = np.clip(attn_np, 1e-6, 1.0)  # Avoid log(0)

    im = ax.imshow(attn_np, cmap='viridis', aspect='equal',
                   norm=LogNorm(vmin=1e-4, vmax=1.0))

    # Highlight specific indices if provided
    if highlight_indices:
        for idx in highlight_indices:
            # Draw lines at these indices
            ax.axhline(y=idx, color='red', linewidth=0.5, alpha=0.7)
            ax.axvline(x=idx, color='red', linewidth=0.5, alpha=0.7)

    ax.set_xlabel('Key Position (Matrix Element)', fontsize=12)
    ax.set_ylabel('Query Position (Matrix Element)', fontsize=12)
    ax.set_title(title, fontsize=14)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, label='Attention Weight')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.pdf', '.png'))
    plt.close()
    print(f"Saved: {save_path}")


def plot_attention_subset(attn_weights, title, save_path, indices):
    """
    Plot attention between specific indices only (zoomed view).

    Args:
        attn_weights: (seq_len, seq_len) attention matrix
        indices: List of indices to show
    """
    # Extract submatrix
    idx = torch.tensor(indices)
    submatrix = attn_weights[idx][:, idx].cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    im = ax.imshow(submatrix, cmap='viridis', aspect='equal')

    # Label with actual indices
    ax.set_xticks(range(len(indices)))
    ax.set_yticks(range(len(indices)))
    ax.set_xticklabels(indices)
    ax.set_yticklabels(indices)

    # Add values in cells
    for i in range(len(indices)):
        for j in range(len(indices)):
            val = submatrix[i, j]
            color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                   color=color, fontsize=9)

    ax.set_xlabel('Key Position', fontsize=12)
    ax.set_ylabel('Query Position', fontsize=12)
    ax.set_title(title, fontsize=14)

    fig.colorbar(im, ax=ax, label='Attention Weight')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.pdf', '.png'))
    plt.close()
    print(f"Saved: {save_path}")


def plot_attention_by_head(attn_weights_all_heads, title_prefix, save_dir, highlight_indices=None):
    """
    Plot attention maps for each head separately.

    Args:
        attn_weights_all_heads: (heads, seq, seq) tensor
    """
    n_heads = attn_weights_all_heads.shape[0]

    fig, axes = plt.subplots(1, n_heads, figsize=(4*n_heads, 4), dpi=150)

    for h in range(n_heads):
        ax = axes[h] if n_heads > 1 else axes
        attn = attn_weights_all_heads[h].cpu().numpy()
        attn = np.clip(attn, 1e-6, 1.0)

        im = ax.imshow(attn, cmap='viridis', aspect='equal',
                       norm=LogNorm(vmin=1e-4, vmax=1.0))

        if highlight_indices:
            for idx in highlight_indices:
                ax.axhline(y=idx, color='red', linewidth=0.5, alpha=0.7)
                ax.axvline(x=idx, color='red', linewidth=0.5, alpha=0.7)

        ax.set_title(f'Head {h+1}', fontsize=12)
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')

    plt.suptitle(title_prefix, fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'{title_prefix.lower().replace(" ", "_")}_by_head.pdf')
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.pdf', '.png'))
    plt.close()
    print(f"Saved: {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Try to load the best transformer model
    # First try v8, then v6
    model_paths = [
        "checkpoints_8/transformer/best.pt",
        "train_v8/checkpoints_8/transformer/best.pt",
        "checkpoints_6/transformer/best.pt",
        "train_v6/checkpoints_6/transformer/best.pt",
    ]

    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break

    if model_path is None:
        print("No trained transformer found. Checking available checkpoints...")
        for path in model_paths:
            print(f"  {path}: {'EXISTS' if os.path.exists(path) else 'NOT FOUND'}")
        print("\nRun training first or specify correct checkpoint path.")
        return

    print(f"Loading model from: {model_path}")

    # Import and load the model
    from train_v8.transformer import TransformerAutoencoder
    from losses.frob import FrobeniusFidelityLoss

    model = TransformerAutoencoder(loss_fn=FrobeniusFidelityLoss())
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Wrap model to capture attention
    wrapper = AttentionCaptureWrapper(model)
    wrapper = wrapper.to(device)
    wrapper.eval()

    # Create output directory
    out_dir = "figures/attention_maps"
    os.makedirs(out_dir, exist_ok=True)

    # === Test 1: GHZ State ===
    print("\n=== GHZ State Analysis ===")
    ghz = create_ghz_state()
    ghz_noisy = add_depolarizing_noise(ghz, p=0.1)
    ghz_input = ghz_noisy.unsqueeze(0).to(device)  # (1, 2, 32, 32)

    with torch.no_grad():
        output = wrapper(ghz_input)

    print(f"Captured {len(wrapper.attention_maps)} attention maps")

    # GHZ state has correlations between indices 0 and 31
    # In flattened 1024 tokens: element (0,0) -> token 0, element (0,31) -> token 31
    # element (31,0) -> token 31*32=992, element (31,31) -> token 31*32+31=1023
    ghz_indices = [0, 31, 992, 1023]  # The 4 corners of the density matrix

    # Plot first encoder layer attention (averaged across heads)
    enc_attn = wrapper.attention_maps[0]['weights'][0]  # (heads, 1024, 1024)
    avg_attn = enc_attn.mean(dim=0)  # Average across heads

    plot_attention_map(
        avg_attn,
        "Encoder Layer 1 Attention (GHZ State, Averaged)",
        f"{out_dir}/ghz_encoder_l1_avg.pdf",
        highlight_indices=ghz_indices
    )

    # Plot per-head attention for first encoder layer
    plot_attention_by_head(
        enc_attn,
        "Encoder L1 GHZ",
        out_dir,
        highlight_indices=ghz_indices
    )

    # Plot zoomed view of GHZ-correlated elements
    plot_attention_subset(
        avg_attn,
        "Attention Between GHZ-Correlated Elements",
        f"{out_dir}/ghz_subset.pdf",
        indices=ghz_indices
    )

    # === Test 2: Random state for comparison ===
    print("\n=== Random State Analysis ===")
    random_state = torch.randn(2, 32, 32)
    # Make it a valid density matrix (roughly)
    random_state = random_state / random_state.norm() * 0.1
    random_input = random_state.unsqueeze(0).to(device)

    with torch.no_grad():
        output = wrapper(random_input)

    enc_attn_random = wrapper.attention_maps[0]['weights'][0]
    avg_attn_random = enc_attn_random.mean(dim=0)

    plot_attention_map(
        avg_attn_random,
        "Encoder Layer 1 Attention (Random State, Averaged)",
        f"{out_dir}/random_encoder_l1_avg.pdf"
    )

    # === Summary statistics ===
    print("\n=== Attention Statistics ===")
    print(f"GHZ state attention at correlated elements:")
    for i, idx_i in enumerate(ghz_indices):
        for j, idx_j in enumerate(ghz_indices):
            val = avg_attn[idx_i, idx_j].item()
            print(f"  ({idx_i}, {idx_j}): {val:.4f}")

    print(f"\nMean attention weight: {avg_attn.mean().item():.6f}")
    print(f"Max attention weight: {avg_attn.max().item():.4f}")
    print(f"Attention at GHZ corners vs mean: {avg_attn[ghz_indices][:, ghz_indices].mean().item() / avg_attn.mean().item():.2f}x")

    # === Analyze all encoder layers ===
    print("\n=== Per-Layer Analysis ===")
    encoder_maps = [m for m in wrapper.attention_maps if m['type'] == 'encoder_self_attn']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)
    axes = axes.flatten()

    for i, attn_map in enumerate(encoder_maps):
        ax = axes[i]
        attn = attn_map['weights'][0].mean(dim=0).cpu().numpy()  # Average across heads

        # Higher contrast: use tighter log scale and plasma colormap
        attn = np.clip(attn, 1e-5, 1.0)
        im = ax.imshow(attn, cmap='plasma', aspect='equal',
                       norm=LogNorm(vmin=1e-3, vmax=0.5))

        ax.set_title(f'Encoder Layer {i+1}', fontsize=12)
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')

        # Compute GHZ attention ratio for this layer
        layer_attn = attn_map['weights'][0].mean(dim=0)
        ghz_attn = layer_attn[ghz_indices][:, ghz_indices].mean().item()
        mean_attn = layer_attn.mean().item()
        ratio = ghz_attn / mean_attn if mean_attn > 0 else 0
        print(f"  Layer {i+1}: GHZ/mean ratio = {ratio:.2f}x")

    plt.suptitle('Encoder Self-Attention Across Layers (GHZ State)', fontsize=14)
    fig.subplots_adjust(right=0.85, hspace=0.25, wspace=0.2)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Attention Weight')
    plt.savefig(f"{out_dir}/ghz_all_encoder_layers.pdf")
    plt.savefig(f"{out_dir}/ghz_all_encoder_layers.png")
    plt.close()
    print(f"Saved: {out_dir}/ghz_all_encoder_layers.pdf")

    # === Cross-attention analysis ===
    print("\n=== Cross-Attention Analysis ===")
    cross_maps = [m for m in wrapper.attention_maps if m['type'] == 'decoder_cross_attn']

    if cross_maps:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)
        axes = axes.flatten()

        for i, attn_map in enumerate(cross_maps):
            ax = axes[i]
            attn = attn_map['weights'][0].mean(dim=0).cpu().numpy()
            attn = np.clip(attn, 1e-6, 1.0)

            im = ax.imshow(attn, cmap='viridis', aspect='equal',
                           norm=LogNorm(vmin=1e-4, vmax=1.0))

            for idx in ghz_indices:
                ax.axhline(y=idx, color='red', linewidth=0.3, alpha=0.5)
                ax.axvline(x=idx, color='red', linewidth=0.3, alpha=0.5)

            ax.set_title(f'Decoder Cross-Attn Layer {i+1}', fontsize=12)
            ax.set_xlabel('Encoder Key')
            ax.set_ylabel('Decoder Query')

        plt.suptitle('Decoder Cross-Attention Across Layers (GHZ State)', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/ghz_cross_attention.pdf")
        plt.savefig(f"{out_dir}/ghz_cross_attention.png")
        plt.close()
        print(f"Saved: {out_dir}/ghz_cross_attention.pdf")

    print(f"\nAll figures saved to: {out_dir}/")


if __name__ == "__main__":
    main()
