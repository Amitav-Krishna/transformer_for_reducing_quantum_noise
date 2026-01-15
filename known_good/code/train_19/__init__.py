"""
train_19: Seed ablation with FIXED data split (seed=42).

Key difference from train_18: The data split is always seed=42,
and only the model initialization seed varies. This properly tests
training reproducibility across random initializations.
"""
