"""
Train Hierarchical Transformer on 5-qubit dataset.

Logs training time to compare against element-wise transformer.
"""

import os
import time
import torch
import csv
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.transformer_hierarchical_5qubit import HierarchicalTransformer5Qubit
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

    # Load dataset
    print("Loading 5-qubit dataset...")
    chunks = load_chunks("dataset_smaller")
    train_chunks, val_chunks, test_chunks = split_chunks(chunks, seed=42)
    print(f"Train: {len(train_chunks)} chunks, Val: {len(val_chunks)} chunks, Test: {len(test_chunks)} chunks")

    # Create model
    model = HierarchicalTransformer5Qubit(loss_fn=FrobeniusFidelityLoss()).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Training config
    EPOCHS = 100
    BATCH_SIZE = 8  # Same as element-wise transformer

    # Create output directories
    os.makedirs("checkpoints_2/hierarchical_transformer_5qubit", exist_ok=True)
    os.makedirs("csvs_2", exist_ok=True)

    # CSV loggers
    train_csv = open("csvs_2/hierarchical_transformer_5qubit_train.csv", "w")
    val_csv = open("csvs_2/hierarchical_transformer_5qubit_val.csv", "w")
    timing_csv = open("csvs_2/hierarchical_transformer_5qubit_timing.csv", "w")

    train_writer = csv.writer(train_csv)
    val_writer = csv.writer(val_csv)
    timing_writer = csv.writer(timing_csv)

    train_writer.writerow(["epoch", "chunk", "batch", "loss"])
    val_writer.writerow(["epoch", "loss"])
    timing_writer.writerow(["epoch", "train_time_seconds", "val_time_seconds", "total_time_seconds"])

    best_val = float("inf")

    print(f"\nStarting training for {EPOCHS} epochs...")

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # Training
        train_start = time.time()
        model.train()

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

                train_writer.writerow([epoch, c_idx, b_idx, float(loss.item())])

        train_time = time.time() - train_start

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

        print(f"Epoch {epoch}: val_loss={val_loss:.6f}, train_time={train_time:.1f}s, val_time={val_time:.1f}s, total={total_time:.1f}s")

        # Save best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "checkpoints_2/hierarchical_transformer_5qubit/best.pt")
            print(f"  → Saved best checkpoint (val_loss={val_loss:.6f})")

    train_csv.close()
    val_csv.close()
    timing_csv.close()

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val:.6f}")


if __name__ == "__main__":
    main()
