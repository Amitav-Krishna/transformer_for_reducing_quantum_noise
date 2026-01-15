"""
Train Wide Hierarchical MLP on 5-qubit dataset (float64).

Capacity control experiment: 5x more parameters than Transformer.

Specs:
- Model: HierarchicalMLP5QubitWide (~5.29M params)
- Data: 32x32 density matrices, float64
- Batch size: 4
- Epochs: 100
- Output: checkpoints_16/mlp_5qubit_wide/
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from train_16.models.mlp_5qubit_wide import HierarchicalMLP5QubitWide
from train_16.train.train_utils import (
    load_5qubit_data,
    train_5qubit_model,
    set_seed,
    DTYPE,
)
from losses.frob import FrobeniusFidelityLoss


def main():
    set_seed(42)

    if not torch.cuda.is_available():
        print("FATAL ERROR: No GPU found. Please fix this.")
        sys.exit(1)
    device = torch.device("cuda")
    print(f"Device: {device}")
    print(f"Dtype: {DTYPE}")

    # Determine output paths
    if os.path.exists("/workspace"):
        checkpoint_dir = "/workspace/checkpoints_16/mlp_5qubit_wide"
        csv_dir = "/workspace/csvs_16"
    else:
        checkpoint_dir = "checkpoints_16/mlp_5qubit_wide"
        csv_dir = "csvs_16"

    # Load data
    train_chunks, val_chunks, test_chunks = load_5qubit_data()

    # Create model
    model = HierarchicalMLP5QubitWide(loss_fn=FrobeniusFidelityLoss())
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Model dtype: {next(model.parameters()).dtype}")

    # Train
    best_val, best_path = train_5qubit_model(
        model=model,
        model_name="mlp_5qubit_wide",
        train_chunks=train_chunks,
        val_chunks=val_chunks,
        device=device,
        checkpoint_dir=checkpoint_dir,
        csv_dir=csv_dir,
    )

    return best_val, best_path


if __name__ == "__main__":
    main()
