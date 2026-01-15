"""
Train Transformer on 8-qubit dataset WITH FROBENIUS NORMALIZATION.

Uses StreamingChunkDataset to avoid OOM on large 8-qubit dataset.

Usage:
    python train_17/train/train_transformer_8q.py
"""

import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import torch
from torch.utils.data import DataLoader, IterableDataset
import random

from train_16.models.transformer_8qubit import HierarchicalTransformer8Qubit
from losses.frob import FrobeniusFidelityLoss

from training_loop.dataset.StreamingChunkDataset import (
    get_chunk_files,
    split_chunk_files,
)
from training_loop.dataset.csv_logger import CSVLogger


def normalize_to_unit(x):
    """Normalize tensor to unit Frobenius norm."""
    norm = x.flatten().norm() + 1e-8
    return x / norm


class StreamingNormalizedDataset(IterableDataset):
    """Streaming dataset with Frobenius normalization."""

    def __init__(self, chunk_files, shuffle=True, seed=42):
        self.chunk_files = chunk_files
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed)
        files = self.chunk_files.copy()
        if self.shuffle:
            rng.shuffle(files)

        for fpath in files:
            blob = torch.load(fpath, weights_only=False, map_location="cpu")
            X = blob["X"].double()
            Y = blob["Y"].double()

            indices = list(range(len(X)))
            if self.shuffle:
                rng.shuffle(indices)

            for idx in indices:
                x = X[idx].permute(2, 0, 1)
                y = Y[idx].permute(2, 0, 1)
                yield normalize_to_unit(x), normalize_to_unit(y)

            del blob, X, Y

    def __len__(self):
        return len(self.chunk_files) * 1000  # Approximate


def evaluate_streaming(model, chunk_files, device, batch_size):
    """Evaluate on streaming dataset."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for fpath in chunk_files:
            blob = torch.load(fpath, weights_only=False, map_location="cpu")
            X = blob["X"].double()
            Y = blob["Y"].double()

            for i in range(0, len(X), batch_size):
                x_batch = X[i : i + batch_size].permute(0, 3, 1, 2)
                y_batch = Y[i : i + batch_size].permute(0, 3, 1, 2)

                # Normalize each sample
                x_norm = torch.stack([normalize_to_unit(x) for x in x_batch])
                y_norm = torch.stack([normalize_to_unit(y) for y in y_batch])

                x_norm, y_norm = x_norm.to(device), y_norm.to(device)
                pred = model(x_norm)
                loss = model.compute_loss(pred, y_norm)
                total_loss += loss.item() * len(x_batch)
                total_samples += len(x_batch)

            del blob, X, Y

    return total_loss / total_samples


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Use streaming dataset
    chunk_files = get_chunk_files("dataset_8qubit_float64")
    train_files, val_files, test_files = split_chunk_files(chunk_files)

    print(f"Train chunks: {len(train_files)}")
    print(f"Val chunks: {len(val_files)}")
    print(f"Test chunks: {len(test_files)}")
    print(f"\nUsing STREAMING + FROBENIUS NORMALIZATION")

    ckpt_dir = os.path.join(base_dir, "checkpoints_17", "transformer_8q")
    os.makedirs(ckpt_dir, exist_ok=True)
    csv_logger = CSVLogger(
        csv_dir=os.path.join(base_dir, "csvs_17"), name="transformer_8q"
    )

    model = HierarchicalTransformer8Qubit(loss_fn=FrobeniusFidelityLoss()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

    EPOCHS = 100
    EARLY_STOP_PATIENCE = 15
    BATCH = 8

    print(f"\n{'=' * 60}")
    print(f"Training Transformer 8-qubit (v17 - Streaming + Frobenius)")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Batch size: {BATCH}")
    print(f"{'=' * 60}\n")

    best_val = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}")
        model.train()

        train_dataset = StreamingNormalizedDataset(
            train_files, shuffle=True, seed=42 + epoch
        )
        train_loader = DataLoader(train_dataset, batch_size=BATCH)

        batch_count = 0
        train_loss_val = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            train_loss = model.compute_loss(pred, y)
            train_loss_val = train_loss.item()

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            batch_count += 1
            if batch_count % 100 == 0:
                print(f"  batch {batch_count} loss = {train_loss_val:.6f}")

            csv_logger.log_train(epoch, 0, batch_count, train_loss_val)

        print(f"  Total batches: {batch_count}, final loss = {train_loss_val:.6f}")

        val_loss = evaluate_streaming(model, val_files, device, BATCH)
        print(f"Validation loss: {val_loss:.6f}")
        csv_logger.log_val(epoch, val_loss)

        if val_loss < best_val:
            best_val = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pt"))
            print(f"Saved BEST checkpoint")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs")

        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"epoch_{epoch}.pt"))

        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping after {epoch} epochs")
            break

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
