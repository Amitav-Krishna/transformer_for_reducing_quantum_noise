"""
5-qubit models for seed ablation experiments.
"""

from train_18.models.transformer_5qubit import HierarchicalTransformer5Qubit
from train_18.models.mlp_5qubit import HierarchicalMLP5Qubit

__all__ = ["HierarchicalTransformer5Qubit", "HierarchicalMLP5Qubit"]
