#!/usr/bin/env python3
"""Check actual attention values for GHZ state analysis."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from train_v8.transformer import TransformerAutoencoder
from losses.frob import FrobeniusFidelityLoss
from figures.attention_maps import AttentionCaptureWrapper, create_ghz_state, add_depolarizing_noise

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TransformerAutoencoder(loss_fn=FrobeniusFidelityLoss())
model.load_state_dict(torch.load('train_v8/checkpoints_8/transformer/best.pt', map_location=device))
model.to(device).eval()

wrapper = AttentionCaptureWrapper(model).to(device)
wrapper.eval()

ghz = create_ghz_state()
ghz_noisy = add_depolarizing_noise(ghz, p=0.1)

with torch.no_grad():
    output = wrapper(ghz_noisy.unsqueeze(0).to(device))

attn = wrapper.attention_maps[0]['weights'][0].mean(dim=0)
attn_to_key0 = attn[:, 0].cpu().numpy()

ghz_indices = [0, 31, 992, 1023]
non_ghz_mask = [i for i in range(1024) if i not in ghz_indices]
non_ghz_attn = attn_to_key0[non_ghz_mask]

print(f'GHZ positions attention: {attn_to_key0[ghz_indices]}')
print(f'Non-GHZ positions:')
print(f'  Mean: {non_ghz_attn.mean():.4f}')
print(f'  Max: {non_ghz_attn.max():.4f}')
print(f'  Median: {sorted(non_ghz_attn)[len(non_ghz_attn)//2]:.4f}')
