"""
Train CNN baseline for density matrix denoising.

Same training setup as train_v8 but with CNN architecture
scaled to ~118k parameters.

Checkpoints saved to: train_12/checkpoints_12/cnn/
Training logs saved to: train_12/csvs_12/

Usage:
    python train_12/cnn/train.py
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from torch.utils.data import DataLoader

from train_12.cnn.cnn import CNNAutoencoder
from losses.frob import FrobeniusFidelityLoss

from training_loop.dataset.load_chunks import load_chunks
from training_loop.dataset.split_chunks import split_chunks
from training_loop.dataset.ChunkDataset import ChunkDataset
from training_loop.dataset.csv_logger import CSVLogger
from training_loop.evaluate_on_chunks import evaluate_on_chunks


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_cnn(train_chunks, val_chunks, device, base_dir):
    """Train CNN model."""
    name = "cnn"

    # Directories
    ckpt_dir = os.path.join(base_dir, "checkpoints_12", "cnn")
    csv_dir = os.path.join(base_dir, "csvs_12")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    # CSV logger
    csv_logger = CSVLogger(csv_dir=csv_dir, name=name)

    # Model + optimizer
    model = CNNAutoencoder(loss_fn=FrobeniusFidelityLoss()).to(device)
    lr = 3e-4
    weight_decay = 1e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Hyperparameters
    EPOCHS = 100
    EARLY_STOP_PATIENCE = 15
    BATCH = 64  # CNN can handle larger batches

    n_params = count_parameters(model)

    print(f"\n{'='*60}")
    print(f"Training CNN Baseline")
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
            # CNN uses same data format as MLP: (B, 2, 32, 32)
            ds = ChunkDataset(X, Y, "cnn")
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
        val_loss = evaluate_on_chunks(model, val_chunks, "cnn", device, BATCH)
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

    # Base directory for outputs
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load and split dataset
    chunks = load_chunks("dataset_smaller")
    train_chunks, val_chunks, test_chunks = split_chunks(chunks)

    print(f"Train chunks: {len(train_chunks)}")
    print(f"Val chunks: {len(val_chunks)}")
    print(f"Test chunks: {len(test_chunks)}")

    # Train CNN
    train_cnn(train_chunks, val_chunks, device, base_dir)

    print("\n" + "="*60)
    print("CNN training complete!")
    print(f"Checkpoints saved to: {base_dir}/checkpoints_12/cnn/")
    print(f"Training logs saved to: {base_dir}/csvs_12/")
    print("="*60)


if __name__ == "__main__":
    main()
