"""
Train Hierarchical MLP on 8-qubit dataset with STREAMING.

Control experiment comparing hierarchical MLP vs hierarchical Transformer.
Uses identical setup to train_hierarchical_8qubit.py for fair comparison.

Expected result: MLP should underperform Transformer, demonstrating that
the attention advantage persists even with hierarchical tokenization.
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

from train_15.mlp_hierarchical_8qubit import HierarchicalMLP8Qubit
from losses.frob import FrobeniusFidelityLoss
from training_loop.dataset.StreamingChunkDataset import (
    StreamingChunkDataset,
    get_chunk_files,
    split_chunk_files,
)


def evaluate_streaming(model, val_files, device, batch_size=4):
    """Evaluate model on validation set using streaming."""
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for fpath in val_files:
            # Load one chunk at a time
            blob = torch.load(fpath, weights_only=False)
            X = blob["X"].float()
            Y = blob["Y"].float()
            del blob

            # Process in batches
            n = len(X)
            for i in range(0, n, batch_size):
                x = X[i : i + batch_size].permute(0, 3, 1, 2).to(device)  # (B, 2, H, W)
                y = Y[i : i + batch_size].permute(0, 3, 1, 2).to(device)

                pred = model(x)
                loss = model.compute_loss(pred, y)
                total_loss += loss.item() * x.shape[0]
                total_samples += x.shape[0]

            del X, Y

    model.train()
    return total_loss / total_samples


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine dataset path (local vs runpod)
    if os.path.exists("/workspace/dataset_8qubit"):
        dataset_dir = "/workspace/dataset_8qubit"
        checkpoint_dir = "/workspace/checkpoints_2/hierarchical_mlp_8qubit"
        csv_dir = "/workspace/csvs_2"
    else:
        # Local fallback (for testing)
        dataset_dir = "dataset_8qubit"
        checkpoint_dir = "checkpoints_2/hierarchical_mlp_8qubit"
        csv_dir = "csvs_2"

    # Get chunk files and split
    print(f"Loading chunk files from {dataset_dir}...")
    chunk_files = get_chunk_files(dataset_dir)
    if len(chunk_files) == 0:
        print(f"ERROR: No chunks found in {dataset_dir}")
        sys.exit(1)

    print(f"Found {len(chunk_files)} chunk files")

    train_files, val_files, test_files = split_chunk_files(chunk_files, seed=42)
    print(
        f"Train: {len(train_files)} chunks, Val: {len(val_files)} chunks, Test: {len(test_files)} chunks"
    )

    # Estimate sample counts (1000 samples per chunk for 8-qubit)
    samples_per_chunk = 1000
    train_samples = len(train_files) * samples_per_chunk
    val_samples = len(val_files) * samples_per_chunk
    print(f"Estimated - Train: {train_samples:,}, Val: {val_samples:,}")

    # Create model
    model = HierarchicalMLP8Qubit(loss_fn=FrobeniusFidelityLoss()).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Optimizer (same as Transformer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    # Training config (same as Transformer)
    EPOCHS = 100
    BATCH_SIZE = 4  # Same as Transformer

    # Create output directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    # CSV loggers
    train_csv_path = os.path.join(csv_dir, "hierarchical_mlp_8qubit_train.csv")
    val_csv_path = os.path.join(csv_dir, "hierarchical_mlp_8qubit_val.csv")
    timing_csv_path = os.path.join(csv_dir, "hierarchical_mlp_8qubit_timing.csv")

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

    print(f"\nStarting training for {EPOCHS} epochs (STREAMING MODE)...")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Logs: {csv_dir}")

    total_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # Training with streaming
        train_start = time.time()
        model.train()

        epoch_losses = []

        # Create streaming dataset for this epoch (reshuffles each epoch)
        train_dataset = StreamingChunkDataset(
            train_files, shuffle=True, samples_per_chunk=samples_per_chunk
        )
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0)

        batch_idx = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = model.compute_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

            # Log every 100 batches to reduce I/O
            if batch_idx % 100 == 0:
                train_writer.writerow(
                    [epoch, batch_idx // 100, batch_idx, float(loss.item())]
                )
            batch_idx += 1

        train_time = time.time() - train_start
        avg_train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0

        # Validation with streaming
        val_start = time.time()
        val_loss = evaluate_streaming(model, val_files, device, BATCH_SIZE)
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
