#!/usr/bin/env python3
"""
Quick overfit test for 8-qubit AXIAL transformer.

Tests if the axial attention model can fit a tiny subset of data.
Compares to the hierarchical model which failed this test.

Key differences from hierarchical:
- 8×8 patches (vs 32×32) → 1024 tokens (vs 64)
- 128 values per patch (vs 2048) → much less compression
- Axial attention (row+col) → O(2×32³) vs O(64²) for global mixing

Run on pod with: python train_16/test_overfit_8qubit_axial.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from train_16.models.transformer_8qubit_axial import AxialTransformer8Qubit
from losses.frob import FrobeniusFidelityLoss


def main():
    # Config
    NUM_TRAIN_CHUNKS = 1  # Just 1 chunk = 1000 samples
    NUM_VAL_CHUNKS = 1  # 1 chunk for validation
    EPOCHS = 10
    BATCH_SIZE = 4  # May need smaller due to more tokens
    LR = 3e-4

    print("=" * 70)
    print("8-QUBIT AXIAL TRANSFORMER OVERFIT TEST")
    print("=" * 70)
    print(f"Train chunks: {NUM_TRAIN_CHUNKS} (~{NUM_TRAIN_CHUNKS * 1000} samples)")
    print(f"Val chunks: {NUM_VAL_CHUNKS} (~{NUM_VAL_CHUNKS * 1000} samples)")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LR}")
    print()
    print("Architecture: 8×8 patches, 1024 tokens, axial attention")
    print()

    # Find dataset
    if os.path.exists("/workspace/dataset_8qubit_float64"):
        dataset_dir = "/workspace/dataset_8qubit_float64"
    elif os.path.exists("dataset_8qubit_float64"):
        dataset_dir = "dataset_8qubit_float64"
    else:
        print("ERROR: No 8-qubit dataset found!")
        print("This script must run on a pod with the dataset.")
        sys.exit(1)

    print(f"Dataset: {dataset_dir}")

    # Get chunk files
    chunk_files = sorted(
        [
            os.path.join(dataset_dir, f)
            for f in os.listdir(dataset_dir)
            if f.endswith(".pt")
        ]
    )
    print(f"Total chunks available: {len(chunk_files)}")

    # Take first few chunks for train/val
    train_files = chunk_files[:NUM_TRAIN_CHUNKS]
    val_files = chunk_files[80 : 80 + NUM_VAL_CHUNKS]  # From val split region

    print(f"Train files: {[os.path.basename(f) for f in train_files]}")
    print(f"Val files: {[os.path.basename(f) for f in val_files]}")
    print()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = AxialTransformer8Qubit(loss_fn=FrobeniusFidelityLoss())
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print()

    # Load training data into memory (it's small enough)
    print("Loading training data...")
    train_X, train_Y = [], []
    for fpath in train_files:
        blob = torch.load(fpath, weights_only=False, map_location="cpu")
        train_X.append(blob["X"])
        train_Y.append(blob["Y"])
        del blob
    train_X = torch.cat(train_X, dim=0)  # (N, 256, 256, 2)
    train_Y = torch.cat(train_Y, dim=0)
    print(f"  Train: {train_X.shape}")

    print("Loading validation data...")
    val_X, val_Y = [], []
    for fpath in val_files:
        blob = torch.load(fpath, weights_only=False, map_location="cpu")
        val_X.append(blob["X"])
        val_Y.append(blob["Y"])
        del blob
    val_X = torch.cat(val_X, dim=0)
    val_Y = torch.cat(val_Y, dim=0)
    print(f"  Val: {val_X.shape}")
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

    # Training loop
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)
    print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | {'Time':>8}")
    print("-" * 50)

    n_train = len(train_X)
    train_losses = []
    val_losses = []

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # Training
        model.train()
        total_loss = 0.0
        n_batches = 0

        # Shuffle
        perm = torch.randperm(n_train)
        train_X_shuffled = train_X[perm]
        train_Y_shuffled = train_Y[perm]

        for i in range(0, n_train, BATCH_SIZE):
            x = train_X_shuffled[i : i + BATCH_SIZE].permute(0, 3, 1, 2).to(device)
            y = train_Y_shuffled[i : i + BATCH_SIZE].permute(0, 3, 1, 2).to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = model.compute_loss(pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / n_batches
        train_losses.append(train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_total = 0.0
            val_batches = 0
            for i in range(0, len(val_X), BATCH_SIZE):
                x = val_X[i : i + BATCH_SIZE].permute(0, 3, 1, 2).to(device)
                y = val_Y[i : i + BATCH_SIZE].permute(0, 3, 1, 2).to(device)
                pred = model(x)
                loss = model.compute_loss(pred, y)
                val_total += loss.item()
                val_batches += 1
            val_loss = val_total / val_batches
            val_losses.append(val_loss)

        elapsed = time.time() - epoch_start
        print(
            f"{epoch:>6} | {train_loss:>12.6f} | {val_loss:>12.6f} | {elapsed:>7.1f}s"
        )

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    # Calculate improvement
    train_improvement = (train_losses[0] - train_losses[-1]) / train_losses[0] * 100
    val_improvement = (val_losses[0] - val_losses[-1]) / val_losses[0] * 100

    print(
        f"Train loss: {train_losses[0]:.6f} -> {train_losses[-1]:.6f} ({train_improvement:.1f}% improvement)"
    )
    print(
        f"Val loss: {val_losses[0]:.6f} -> {val_losses[-1]:.6f} ({val_improvement:.1f}% improvement)"
    )
    print()

    # Interpretation
    if train_losses[-1] < 0.7:
        print("SUCCESS: Model can fit training data well (train loss < 0.7)")
        print("The axial attention architecture has appropriate inductive bias.")
        print("Next step: full training with proper hyperparameters.")
    elif train_losses[-1] < 0.85:
        print("PARTIAL SUCCESS: Model is learning but slowly")
        print("May need: more epochs, higher LR, or architecture tweaks.")
    else:
        print("FAILURE: Model cannot fit even tiny training set")
        print("The architecture's inductive bias is still wrong.")

    print()
    print("Comparison to hierarchical (32×32 patches):")
    print("  Hierarchical final train loss: 0.911 (FAILED)")
    print(f"  Axial final train loss: {train_losses[-1]:.3f}")


if __name__ == "__main__":
    main()
