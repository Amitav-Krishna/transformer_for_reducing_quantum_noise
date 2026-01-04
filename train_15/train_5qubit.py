"""
Train Hierarchical MLP on 5-qubit dataset.

Control experiment comparing hierarchical MLP vs hierarchical Transformer.
Uses same setup as train_hierarchical_5qubit.py for fair comparison.

Expected result: MLP should underperform Transformer, demonstrating that
the attention advantage persists with hierarchical tokenization.
"""

import os
import sys
import time
import torch
import csv
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_15.mlp_hierarchical_5qubit import HierarchicalMLP5Qubit
from losses.frob import FrobeniusFidelityLoss
from training_loop.dataset.load_chunks import load_chunks
from training_loop.dataset.split_chunks import split_chunks
from training_loop.dataset.ChunkDataset import ChunkDataset


def evaluate(model, chunks, device, batch_size=8):
    """Evaluate model on validation/test chunks."""
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for X, Y in chunks:
            ds = ChunkDataset(X, Y, mode="transformer")
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

            for x, y in loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = model.compute_loss(pred, y)
                total_loss += loss.item() * x.shape[0]
                total_samples += x.shape[0]

    model.train()
    return total_loss / total_samples


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine dataset path (local vs runpod)
    if os.path.exists("/workspace/dataset_smaller"):
        dataset_dir = "/workspace/dataset_smaller"
        checkpoint_dir = "/workspace/checkpoints_2/hierarchical_mlp_5qubit"
        csv_dir = "/workspace/csvs_2"
    else:
        dataset_dir = "dataset_smaller"
        checkpoint_dir = "checkpoints_2/hierarchical_mlp_5qubit"
        csv_dir = "csvs_2"

    # Load dataset
    print(f"Loading dataset from {dataset_dir}...")
    chunks = load_chunks(dataset_dir)
    if len(chunks) == 0:
        print(f"ERROR: No chunks found in {dataset_dir}")
        sys.exit(1)

    train_chunks, val_chunks, test_chunks = split_chunks(chunks, seed=42)
    print(
        f"Train: {len(train_chunks)} chunks, Val: {len(val_chunks)} chunks, Test: {len(test_chunks)} chunks"
    )

    # Count samples
    train_samples = sum(X.shape[0] for X, Y in train_chunks)
    val_samples = sum(X.shape[0] for X, Y in val_chunks)
    print(f"Train samples: {train_samples:,}, Val samples: {val_samples:,}")

    # Create model
    model = HierarchicalMLP5Qubit(loss_fn=FrobeniusFidelityLoss()).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Optimizer (same as Transformer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    # Training config (same as Transformer)
    EPOCHS = 100
    BATCH_SIZE = 8

    # Create output directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    # CSV loggers
    train_csv_path = os.path.join(csv_dir, "hierarchical_mlp_5qubit_train.csv")
    val_csv_path = os.path.join(csv_dir, "hierarchical_mlp_5qubit_val.csv")
    timing_csv_path = os.path.join(csv_dir, "hierarchical_mlp_5qubit_timing.csv")

    train_csv = open(train_csv_path, "w")
    val_csv = open(val_csv_path, "w")
    timing_csv = open(timing_csv_path, "w")

    train_writer = csv.writer(train_csv)
    val_writer = csv.writer(val_csv)
    timing_writer = csv.writer(timing_csv)

    train_writer.writerow(["epoch", "chunk", "batch", "loss"])
    val_writer.writerow(["epoch", "loss"])
    timing_writer.writerow(
        ["epoch", "train_time_seconds", "val_time_seconds", "total_time_seconds"]
    )

    best_val = float("inf")

    print(f"\nStarting training for {EPOCHS} epochs...")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Logs: {csv_dir}")

    total_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # Training
        train_start = time.time()
        model.train()

        epoch_losses = []
        for c_idx, (X, Y) in enumerate(train_chunks):
            ds = ChunkDataset(X, Y, mode="transformer")
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

            for b_idx, (x, y) in enumerate(loader):
                x, y = x.to(device), y.to(device)

                pred = model(x)
                loss = model.compute_loss(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            # Log final batch loss per chunk
            if epoch_losses:
                train_writer.writerow([epoch, c_idx, b_idx, float(epoch_losses[-1])])

        train_time = time.time() - train_start
        avg_train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0

        # Validation
        val_start = time.time()
        val_loss = evaluate(model, val_chunks, device, BATCH_SIZE)
        val_time = time.time() - val_start

        total_time = time.time() - epoch_start

        # Log
        val_writer.writerow([epoch, float(val_loss)])
        timing_writer.writerow([epoch, train_time, val_time, total_time])
        train_csv.flush()
        val_csv.flush()
        timing_csv.flush()

        print(
            f"Epoch {epoch}/{EPOCHS} | Train: {avg_train_loss:.6f} | Val: {val_loss:.6f} | "
            f"Time: {total_time:.1f}s (train: {train_time:.1f}s, val: {val_time:.1f}s)"
        )

        # Save best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best.pt"))
            print(f"  -> Saved best checkpoint (val_loss={val_loss:.6f})")

        # Save periodic checkpoints
        if epoch % 10 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt"),
            )

    train_csv.close()
    val_csv.close()
    timing_csv.close()

    total_elapsed = time.time() - total_start
    print(f"\nTraining complete!")
    print(f"Total time: {total_elapsed / 3600:.2f} hours")
    print(f"Best validation loss: {best_val:.6f}")
    print(f"Best checkpoint: {checkpoint_dir}/best.pt")


if __name__ == "__main__":
    main()
