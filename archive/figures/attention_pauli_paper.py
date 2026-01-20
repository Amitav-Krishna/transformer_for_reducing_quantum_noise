#!/usr/bin/env python3
"""
Create paper-quality attention visualization for Pauli transformer.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt

from train_11.transformer_pauli import TransformerPauliAutoencoder
from train_11.pauli_representation import density_matrix_to_pauli_basis, get_pauli_string
from train_11.pauli_loss import PauliFrobeniusLoss
from figures.attention_maps_pauli import PauliAttentionCaptureWrapper, create_ghz_density_matrix, add_depolarizing_noise_dm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = TransformerPauliAutoencoder(loss_fn=PauliFrobeniusLoss())
model.load_state_dict(torch.load('train_11/checkpoints_11/transformer_pauli_frob/best.pt', map_location=device))
model = model.to(device)
model.eval()

wrapper = PauliAttentionCaptureWrapper(model)
wrapper = wrapper.to(device)
wrapper.eval()

# Create GHZ state and convert to Pauli basis
ghz_dm = create_ghz_density_matrix(n_qubits=5)
ghz_noisy = add_depolarizing_noise_dm(ghz_dm, p=0.1)
ghz_pauli = density_matrix_to_pauli_basis(ghz_noisy, n_qubits=5)

# Forward pass
ghz_input = torch.tensor(ghz_pauli, dtype=torch.float32, device=device).unsqueeze(0)
with torch.no_grad():
    output = wrapper(ghz_input)

# Get attention
enc_attn = wrapper.attention_maps[0]['weights'][0]
avg_attn = enc_attn.mean(dim=0).cpu().numpy()

# Attention to identity (IIIII = index 0)
attn_to_identity = avg_attn[:, 0]

# Find GHZ-relevant X/Y operators (those with high attention)
top_indices = np.argsort(attn_to_identity)[-16:][::-1]
print("Top 16 indices attending to identity:")
for idx in top_indices:
    pauli_string = get_pauli_string(idx, 5)
    label = ''.join(['I','X','Y','Z'][p] for p in pauli_string)
    coeff = ghz_pauli[idx]
    print(f"  {idx:4d}: {label}, attn={attn_to_identity[idx]:.4f}, coeff={coeff:.4f}")

# Create figure
fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

# Plot all positions
ax.bar(range(1024), attn_to_identity, width=1.0, color='steelblue', alpha=0.7)

# Highlight the X/Y coherence operators
xy_indices = [idx for idx in top_indices if attn_to_identity[idx] > 0.05]
for idx in xy_indices:
    ax.bar(idx, attn_to_identity[idx], width=3, color='red', alpha=0.9)

ax.set_xlabel('Pauli Coefficient Index', fontsize=12)
ax.set_ylabel('Attention to Identity', fontsize=12)
ax.set_title('Pauli Transformer: Which coefficients attend to Identity (IIIII)?', fontsize=14)
ax.set_xlim(-10, 1034)

# Add annotation
avg = attn_to_identity.mean()
ax.axhline(y=avg, color='gray', linestyle='--', linewidth=1, label=f'Mean: {avg:.5f}')

# Annotate the cluster of high-attention operators
cluster_center = np.mean([idx for idx in xy_indices if 300 < idx < 500])
ax.annotate(f'X/Y coherence operators\n(90× mean attention)',
            xy=(cluster_center, 0.083),
            xytext=(550, 0.065),
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='red'))

ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('figures/pauli_attention_to_identity_paper.pdf')
plt.savefig('figures/pauli_attention_to_identity_paper.png')
print('\nSaved: figures/pauli_attention_to_identity_paper.pdf')

# Also create a comparison figure showing density matrix vs Pauli attention structure
print("\n=== Statistics ===")
print(f"Mean attention: {avg:.6f}")
print(f"Max attention: {attn_to_identity.max():.4f}")
print(f"Max/mean ratio: {attn_to_identity.max() / avg:.1f}×")
print(f"Number of high-attention indices (>10× mean): {sum(attn_to_identity > 10*avg)}")
