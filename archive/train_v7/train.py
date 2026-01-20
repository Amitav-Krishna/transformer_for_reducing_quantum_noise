"""
Train Residual MLP (v7).

The residual MLP learns corrections to the input rather than
full reconstruction, preserving information through skip connection.

Checkpoints saved to: checkpoints_7/
Training logs saved to: csvs_7/

Usage:
    python train_v7/train.py
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from train_v7.mlp import MLPResidualAutoencoder
from losses.frob import FrobeniusFidelityLoss

from training_loop.dataset.load_chunks import load_chunks
from training_loop.dataset.split_chunks import split_chunks
from training_loop.dataset.ChunkDataset import ChunkDataset
from training_loop.dataset.csv_logger import CSVLogger
from training_loop.evaluate_on_chunks import evaluate_on_chunks


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_mlp(train_chunks, val_chunks, device):
    """Train the residual MLP."""

    # Directories
    ckpt_dir = "checkpoints_7/mlp"
    os.makedirs(ckpt_dir, exist_ok=True)

    # CSV logger
    csv_logger = CSVLogger(csv_dir="csvs_7", name="mlp")

    # Model + optimizer
    model = MLPResidualAutoencoder(loss_fn=FrobeniusFidelityLoss()).to(device)
    lr = 3e-4
    weight_decay = 1e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Hyperparameters
    EPOCHS = 100
    EARLY_STOP_PATIENCE = 15
    BATCH = 64

    n_params = count_parameters(model)

    print(f"\n{'='*60}")
    print(f"Training Residual MLP")
    print(f"Architecture: input + correction(input)")
    print(f"Parameters: {n_params:,}")
    print(f"Batch size: {BATCH} | LR: {lr} | Weight decay: {weight_decay}")
    print(f"Early stopping patience: {EARLY_STOP_PATIENCE}")
    print(f"{'='*60}\n")

    best_val = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}")

        # Training over all chunks
        model.train()
        for c_idx, (X, Y) in enumerate(train_chunks):
            ds = ChunkDataset(X, Y, "mlp")
            loader = DataLoader(ds, batch_size=BATCH, shuffle=True)

            for b_idx, (x, y) in enumerate(loader):
                x, y = x.to(device), y.to(device)

                pred = model(x)
                train_loss = model.compute_loss(pred, y)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                csv_logger.log_train(epoch, c_idx, b_idx, float(train_loss.item()))

            print(f"  chunk {c_idx} final batch loss = {train_loss.item():.6f}")

        # Validation
        val_loss = evaluate_on_chunks(model, val_chunks, "mlp", device, BATCH)
        print(f"Validation loss: {val_loss:.6f}")

        csv_logger.log_val(epoch, float(val_loss))

        # Save best checkpoint and track early stopping
        if val_loss < best_val:
            best_val = val_loss
            epochs_without_improvement = 0
            best_path = os.path.join(ckpt_dir, "best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"Saved BEST checkpoint -> {best_path}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs")

        # Save epoch checkpoint
        epoch_path = os.path.join(ckpt_dir, f"epoch_{epoch}.pt")
        torch.save(model.state_dict(), epoch_path)
        print(f"Saved checkpoint -> {epoch_path}")

        # Early stopping
        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            print(f"Best validation loss: {best_val:.6f}")
            break


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and split dataset
    chunks = load_chunks("dataset_smaller")
    train_chunks, val_chunks, test_chunks = split_chunks(chunks)

    print(f"Train chunks: {len(train_chunks)}")
    print(f"Val chunks: {len(val_chunks)}")
    print(f"Test chunks: {len(test_chunks)}")

    # Create output directories
    os.makedirs("checkpoints_7", exist_ok=True)
    os.makedirs("csvs_7", exist_ok=True)

    # Train
    train_mlp(train_chunks, val_chunks, device)

    print("\n" + "="*60)
    print("Training complete!")
    print("Checkpoints saved to: checkpoints_7/")
    print("Training logs saved to: csvs_7/")
    print("="*60)


if __name__ == "__main__":
    main()
