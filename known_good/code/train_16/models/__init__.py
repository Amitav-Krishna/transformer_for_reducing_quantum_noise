"""
Float64 models for quantum density matrix denoising.

All models in this module use torch.float64 for weights and computations.
"""

from train_16.models.transformer_5qubit import HierarchicalTransformer5Qubit
from train_16.models.transformer_8qubit import HierarchicalTransformer8Qubit
from train_16.models.mlp_5qubit import HierarchicalMLP5Qubit
from train_16.models.mlp_5qubit_wide import HierarchicalMLP5QubitWide
from train_16.models.mlp_5qubit_deep import HierarchicalMLP5QubitDeep
from train_16.models.mlp_8qubit import HierarchicalMLP8Qubit

__all__ = [
    "HierarchicalTransformer5Qubit",
    "HierarchicalTransformer8Qubit",
    "HierarchicalMLP5Qubit",
    "HierarchicalMLP5QubitWide",
    "HierarchicalMLP5QubitDeep",
    "HierarchicalMLP8Qubit",
]
