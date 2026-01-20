"""
train_18: Seed ablation experiments for 5-qubit models.

Retrains Transformer and MLP with different random seeds (100, 200)
to verify reproducibility and measure variance in final Uhlmann fidelity.

Original experiments used seed=42. This directory tests seeds 100 and 200.
"""
