#!/usr/bin/env python3
"""
Visualize attention maps from the Pauli-basis Transformer on entangled states.

Shows which Pauli coefficients attend to which other coefficients,
demonstrating that the model learns algebraic entanglement structure
even without spatial layout.

Usage:
    python figures/attention_maps_pauli.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from train_11.transformer_pauli import TransformerPauliAutoencoder
from train_11.pauli_representation import density_matrix_to_pauli_basis, get_pauli_string
from train_11.pauli_loss import PauliFrobeniusLoss


class PauliAttentionCaptureWrapper(nn.Module):
    """Wrapper that captures attention weights from Pauli TransformerEncoder."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.attention_maps = []

    def forward(self, x):
        """Forward pass that captures attention weights."""
        B = x.shape[0]

        # Embed each Pauli coefficient
        x_embed = self.model.token_embed(x.unsqueeze(-1))  # (B, 1024, embed_dim)

        # Manually run through encoder layers to capture attention
        self.attention_maps = []
        enc = x_embed
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

            # Complete the layer forward pass
            x2 = layer.norm1(enc + layer.dropout1(attn_out))
            ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(x2))))
            enc = layer.norm2(x2 + layer.dropout2(ff_out))

        # Bottleneck
        h_bottleneck = torch.stack([
            self.model.bottleneck(enc[:, i, :]) for i in range(self.model.n_tokens)
        ], dim=1)

        # Decoder with attention capture
        dec = h_bottleneck
        for i, layer in enumerate(self.model.decoder.layers):
            attn_out, attn_weights = layer.self_attn(
                dec, dec, dec,
                need_weights=True,
                average_attn_weights=False
            )
            self.attention_maps.append({
                'layer': i,
                'type': 'decoder_self_attn',
                'weights': attn_weights.detach()
            })

            x2 = layer.norm1(dec + layer.dropout1(attn_out))
            ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(x2))))
            dec = layer.norm2(x2 + layer.dropout2(ff_out))

        out = self.model.output_proj(dec).squeeze(-1)
        return out


def create_ghz_density_matrix(n_qubits=5):
    """
    Create a GHZ state density matrix: |GHZ> = (|00...0> + |11...1>) / sqrt(2)

    Returns: (32, 32) complex array
    """
    dim = 2 ** n_qubits
    rho = np.zeros((dim, dim), dtype=complex)

    # |GHZ><GHZ| = 0.5 * (|0><0| + |0><1| + |1><0| + |1><1|)
    # where |0> = |00000> and |1> = |11111>
    rho[0, 0] = 0.5      # |00000><00000|
    rho[dim-1, dim-1] = 0.5  # |11111><11111|
    rho[0, dim-1] = 0.5  # |00000><11111| (coherence)
    rho[dim-1, 0] = 0.5  # |11111><00000| (coherence)

    return rho


def add_depolarizing_noise_dm(rho, p=0.1):
    """Add depolarizing noise to a density matrix."""
    dim = rho.shape[0]
    identity = np.eye(dim, dtype=complex) / dim
    return (1 - p) * rho + p * identity


def get_ghz_relevant_pauli_indices(n_qubits=5):
    """
    Find Pauli indices that are non-zero for the GHZ state.

    For GHZ = (|00...0> + |11...1>) / sqrt(2):
    - All Z-strings have expectation value 1 (e.g., ZZZZZ, ZZZZI, ...)
    - X-strings of even weight have expectation value 1 (e.g., XXXXX)
    - Y-strings of even weight (with specific phase) are also non-zero

    Returns indices and their Pauli string representations.
    """
    important_indices = []
    labels = []

    for i in range(4**n_qubits):
        pauli_string = get_pauli_string(i, n_qubits)

        # Count X, Y, Z occurrences
        n_x = sum(1 for p in pauli_string if p == 1)  # X
        n_y = sum(1 for p in pauli_string if p == 2)  # Y
        n_z = sum(1 for p in pauli_string if p == 3)  # Z
        n_i = sum(1 for p in pauli_string if p == 0)  # I

        # GHZ state expectations:
        # - Pure Z strings: all have |<Z...Z>| > 0
        # - X strings with all X's: <XXXXX> = 1
        # - Mixed X/Z or Y involvement: varies

        # Key patterns for GHZ:
        # 1. Identity (index 0)
        # 2. All Z's and any subset of I's
        # 3. All X's (XXXXX)
        # 4. All Y's with even number (for 5 qubits, YYYYY has odd parity)

        if n_y == 0 and n_x == 0:  # Pure Z/I strings
            important_indices.append(i)
            pauli_chars = ['I', 'X', 'Y', 'Z']
            label = ''.join(pauli_chars[p] for p in pauli_string)
            labels.append(label)
        elif n_z == 0 and n_y == 0 and n_x == n_qubits:  # XXXXX
            important_indices.append(i)
            labels.append('XXXXX')

    return important_indices, labels


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Pauli transformer
    model_path = "train_11/checkpoints_11/transformer_pauli_frob/best.pt"
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    print(f"Loading model from: {model_path}")
    model = TransformerPauliAutoencoder(loss_fn=PauliFrobeniusLoss())
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Wrap to capture attention
    wrapper = PauliAttentionCaptureWrapper(model)
    wrapper = wrapper.to(device)
    wrapper.eval()

    # Create output directory
    out_dir = "figures/attention_maps_pauli"
    os.makedirs(out_dir, exist_ok=True)

    # Create GHZ state and convert to Pauli basis
    print("\n=== GHZ State in Pauli Basis ===")
    ghz_dm = create_ghz_density_matrix(n_qubits=5)
    ghz_noisy = add_depolarizing_noise_dm(ghz_dm, p=0.1)

    # Convert to format expected by density_matrix_to_pauli_basis
    ghz_pauli = density_matrix_to_pauli_basis(ghz_noisy, n_qubits=5)
    print(f"Pauli coefficients shape: {ghz_pauli.shape}")
    print(f"Pauli range: [{ghz_pauli.min():.4f}, {ghz_pauli.max():.4f}]")

    # Find GHZ-relevant Pauli indices
    ghz_indices, ghz_labels = get_ghz_relevant_pauli_indices(n_qubits=5)
    print(f"\nGHZ-relevant Pauli operators ({len(ghz_indices)} total):")
    for idx, label in zip(ghz_indices[:10], ghz_labels[:10]):
        print(f"  Index {idx:4d}: {label}, coeff = {ghz_pauli[idx]:.4f}")
    print("  ...")

    # Find indices with largest coefficients
    top_k = 20
    top_indices = np.argsort(np.abs(ghz_pauli))[-top_k:][::-1]
    print(f"\nTop {top_k} Pauli coefficients by magnitude:")
    for idx in top_indices:
        pauli_string = get_pauli_string(idx, 5)
        pauli_chars = ['I', 'X', 'Y', 'Z']
        label = ''.join(pauli_chars[p] for p in pauli_string)
        print(f"  Index {idx:4d}: {label}, coeff = {ghz_pauli[idx]:.4f}")

    # Forward pass to capture attention
    ghz_input = torch.tensor(ghz_pauli, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        output = wrapper(ghz_input)

    print(f"\nCaptured {len(wrapper.attention_maps)} attention maps")

    # Get first encoder layer attention (averaged across heads)
    enc_attn = wrapper.attention_maps[0]['weights'][0]  # (heads, 1024, 1024)
    avg_attn = enc_attn.mean(dim=0).cpu().numpy()  # Average across heads

    # Plot attention to identity (index 0 = IIIII)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    attn_to_identity = avg_attn[:, 0]  # All queries' attention to identity
    ax.bar(range(1024), attn_to_identity, width=1.0, color='steelblue', alpha=0.7)

    # Highlight GHZ-relevant indices
    for idx in ghz_indices[:32]:  # Just Z-strings (first 32)
        ax.bar(idx, attn_to_identity[idx], width=2, color='red', alpha=0.9)

    ax.set_xlabel('Query Position (Pauli coefficient index)', fontsize=12)
    ax.set_ylabel('Attention to Identity (IIIII)', fontsize=12)
    ax.set_title('Pauli Transformer: Attention to Identity Coefficient', fontsize=14)
    ax.set_xlim(-10, 1034)

    avg = attn_to_identity.mean()
    ax.axhline(y=avg, color='gray', linestyle='--', linewidth=1, label=f'Average: {avg:.5f}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{out_dir}/pauli_attention_to_identity.pdf')
    plt.savefig(f'{out_dir}/pauli_attention_to_identity.png')
    print(f'Saved: {out_dir}/pauli_attention_to_identity.pdf')
    plt.close()

    # Plot attention FROM identity to all keys
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    attn_from_identity = avg_attn[0, :]  # Identity's attention to all keys
    ax.bar(range(1024), attn_from_identity, width=1.0, color='steelblue', alpha=0.7)

    # Highlight top attention targets
    top_targets = np.argsort(attn_from_identity)[-10:]
    for idx in top_targets:
        ax.bar(idx, attn_from_identity[idx], width=2, color='red', alpha=0.9)

    ax.set_xlabel('Key Position (Pauli coefficient index)', fontsize=12)
    ax.set_ylabel('Attention from Identity (IIIII)', fontsize=12)
    ax.set_title('Pauli Transformer: Where does Identity attend?', fontsize=14)
    ax.set_xlim(-10, 1034)

    plt.tight_layout()
    plt.savefig(f'{out_dir}/pauli_attention_from_identity.pdf')
    plt.savefig(f'{out_dir}/pauli_attention_from_identity.png')
    print(f'Saved: {out_dir}/pauli_attention_from_identity.pdf')
    plt.close()

    # Plot attention between high-magnitude coefficients
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    # Focus on XXXXX index (should be strongly correlated for GHZ)
    xxxxx_idx = 1 + 4 + 16 + 64 + 256  # Each X is 1 in base-4, so XXXXX = 1+4+16+64+256 = 341
    print(f"\nXXXXX index: {xxxxx_idx}")
    pauli_string = get_pauli_string(xxxxx_idx, 5)
    verification = ''.join(['I','X','Y','Z'][p] for p in pauli_string)
    print(f"  Verification: {verification}")

    attn_to_xxxxx = avg_attn[:, xxxxx_idx]
    ax.bar(range(1024), attn_to_xxxxx, width=1.0, color='steelblue', alpha=0.7)

    # Highlight Z-string indices
    for idx in ghz_indices[:32]:
        ax.bar(idx, attn_to_xxxxx[idx], width=2, color='red', alpha=0.9)

    ax.set_xlabel('Query Position (Pauli coefficient index)', fontsize=12)
    ax.set_ylabel(f'Attention to XXXXX (index {xxxxx_idx})', fontsize=12)
    ax.set_title('Pauli Transformer: Attention to XXXXX Coefficient', fontsize=14)
    ax.set_xlim(-10, 1034)

    plt.tight_layout()
    plt.savefig(f'{out_dir}/pauli_attention_to_xxxxx.pdf')
    plt.savefig(f'{out_dir}/pauli_attention_to_xxxxx.png')
    print(f'Saved: {out_dir}/pauli_attention_to_xxxxx.pdf')
    plt.close()

    # Compare attention patterns to density matrix transformer
    # Show attention heatmap for subset of important indices
    important_subset = sorted(set(ghz_indices[:16] + list(top_indices)))[:20]
    subset_attn = avg_attn[np.ix_(important_subset, important_subset)]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    im = ax.imshow(subset_attn, cmap='viridis', aspect='equal')

    # Labels
    subset_labels = []
    for idx in important_subset:
        pauli_string = get_pauli_string(idx, 5)
        pauli_chars = ['I', 'X', 'Y', 'Z']
        subset_labels.append(''.join(pauli_chars[p] for p in pauli_string))

    ax.set_xticks(range(len(important_subset)))
    ax.set_yticks(range(len(important_subset)))
    ax.set_xticklabels(subset_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(subset_labels, fontsize=8)

    ax.set_xlabel('Key (Pauli operator)', fontsize=12)
    ax.set_ylabel('Query (Pauli operator)', fontsize=12)
    ax.set_title('Attention Between GHZ-Relevant Pauli Coefficients', fontsize=14)

    plt.colorbar(im, ax=ax, label='Attention Weight')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/pauli_attention_subset.pdf')
    plt.savefig(f'{out_dir}/pauli_attention_subset.png')
    print(f'Saved: {out_dir}/pauli_attention_subset.pdf')
    plt.close()

    # Summary statistics
    print("\n=== Attention Statistics ===")
    print(f"Mean attention weight: {avg_attn.mean():.6f}")
    print(f"Max attention weight: {avg_attn.max():.4f}")

    # Attention between Z-strings vs random pairs
    z_string_indices = ghz_indices[:32]  # Pure Z/I strings
    z_attn = avg_attn[np.ix_(z_string_indices, z_string_indices)].mean()
    print(f"Mean attention within Z-strings: {z_attn:.4f}")
    print(f"Z-string attention vs mean: {z_attn / avg_attn.mean():.2f}x")

    print(f"\nAll figures saved to: {out_dir}/")


if __name__ == "__main__":
    main()
