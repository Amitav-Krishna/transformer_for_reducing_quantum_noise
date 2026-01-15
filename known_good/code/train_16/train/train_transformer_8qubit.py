"""
Train Hierarchical Transformer on 8-qubit dataset (float64, streaming).

Specs:
- Model: HierarchicalTransformer8Qubit (~1.61M params)
- Data: 256x256 density matrices, float64, streaming
- Batch size: 4
- Epochs: 100
- Output: checkpoints_16/transformer_8qubit/
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from train_16.models.transformer_8qubit import HierarchicalTransformer8Qubit
from train_16.train.train_utils import (
    load_8qubit_file_lists,
    train_8qubit_model,
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
        checkpoint_dir = "/workspace/checkpoints_16/transformer_8qubit"
        csv_dir = "/workspace/csvs_16"
    else:
        checkpoint_dir = "checkpoints_16/transformer_8qubit"
        csv_dir = "csvs_16"

    # Load file lists (streaming)
    train_files, val_files, test_files = load_8qubit_file_lists()

    # Create model
    model = HierarchicalTransformer8Qubit(loss_fn=FrobeniusFidelityLoss())
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Model dtype: {next(model.parameters()).dtype}")

    # Train
    best_val, best_path = train_8qubit_model(
        model=model,
        model_name="transformer_8qubit",
        train_files=train_files,
        val_files=val_files,
        device=device,
        checkpoint_dir=checkpoint_dir,
        csv_dir=csv_dir,
    )

    return best_val, best_path


if __name__ == "__main__":
    main()
