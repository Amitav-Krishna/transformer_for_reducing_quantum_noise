#!/usr/bin/env python3
"""
Create a simple, clear attention visualization.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt

from train_v8.transformer import TransformerAutoencoder
from losses.frob import FrobeniusFidelityLoss
from figures.attention_maps import AttentionCaptureWrapper, create_ghz_state, add_depolarizing_noise

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = TransformerAutoencoder(loss_fn=FrobeniusFidelityLoss())
model.load_state_dict(torch.load('train_v8/checkpoints_8/transformer/best.pt', map_location=device))
model = model.to(device)
model.eval()

wrapper = AttentionCaptureWrapper(model)
wrapper = wrapper.to(device)
wrapper.eval()

# Create GHZ state
ghz = create_ghz_state()
ghz_noisy = add_depolarizing_noise(ghz, p=0.1)
ghz_input = ghz_noisy.unsqueeze(0).to(device)

with torch.no_grad():
    output = wrapper(ghz_input)

# Get encoder layer 1 attention
attn = wrapper.attention_maps[0]['weights'][0].mean(dim=0).cpu().numpy()

# Create figure: bar chart showing mean attention TO each key position
fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

# Compute mean attention received by each key (column)
mean_attn_per_key = attn.mean(axis=0)  # Average across all queries

# GHZ positions
ghz_keys = [0, 31, 992, 1023]

# Plot all positions
ax.bar(range(1024), mean_attn_per_key, width=1.0, color='steelblue', alpha=0.7)

# Highlight GHZ positions
for k in ghz_keys:
    ax.bar(k, mean_attn_per_key[k], width=3, color='red', alpha=0.9)

ax.set_xlabel('Key Position (density matrix element)', fontsize=12)
ax.set_ylabel('Mean Attention Received', fontsize=12)
ax.set_title('Which matrix elements receive the most attention?', fontsize=14)
ax.set_xlim(-10, 1034)

# Add annotation
ax.annotate(f'Token 0: {mean_attn_per_key[0]:.4f}\n(12x average)',
            xy=(0, mean_attn_per_key[0]),
            xytext=(100, mean_attn_per_key[0] * 0.8),
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='red'))

avg = mean_attn_per_key.mean()
ax.axhline(y=avg, color='gray', linestyle='--', linewidth=1, label=f'Average: {avg:.5f}')
ax.legend()

plt.tight_layout()
plt.savefig('figures/attention_to_keys.pdf')
plt.savefig('figures/attention_to_keys.png')
print('Saved: figures/attention_to_keys.pdf')

# Second figure: which queries attend most to key 0?
fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

attn_to_key0 = attn[:, 0]  # All queries' attention to key 0

ax.bar(range(1024), attn_to_key0, width=1.0, color='steelblue', alpha=0.7)

# Highlight GHZ query positions
for q in ghz_keys:
    ax.bar(q, attn_to_key0[q], width=3, color='red', alpha=0.9)

ax.set_xlabel('Query Position (density matrix element)', fontsize=12)
ax.set_ylabel('Attention to Key 0', fontsize=12)
ax.set_title('Which matrix elements attend most strongly to element (0,0)?', fontsize=14)
ax.set_xlim(-10, 1034)

# Add annotations for the 4 GHZ positions
ax.annotate(f'GHZ elements\n(attention = 0.5)',
            xy=(31, 0.5),
            xytext=(150, 0.4),
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='red'))

plt.tight_layout()
plt.savefig('figures/attention_from_queries_to_key0.pdf')
plt.savefig('figures/attention_from_queries_to_key0.png')
print('Saved: figures/attention_from_queries_to_key0.pdf')
