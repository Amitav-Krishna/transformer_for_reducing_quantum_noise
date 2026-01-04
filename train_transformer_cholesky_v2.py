"""
Train ONLY the fixed Transformer Cholesky model.

The original transformer collapsed because:
1. Degenerate decoder: dec = self.decoder(enc, enc)
2. Per-row output projection prevented global Cholesky coordination
3. CLS token was discarded

This version uses the fixed TransformerCholeskyAutoencoder which:
- Uses learnable decoder query instead of degenerate self-attention
- Uses CLS token for global prediction of all 1024 Cholesky params
- Enables global coordination needed for valid density matrices

Checkpoints saved to: checkpoints_3/transformer_cholesky_v2/
Training logs saved to: csvs_3/transformer_cholesky_v2.csv

Usage:
    python train_transformer_cholesky_v2.py
"""

import os
import torch
from torch.utils.data import DataLoader

from models_3.transformer_cholesky import TransformerCholeskyAutoencoder
from losses.frob import FrobeniusFidelityLoss

from training_loop.dataset.load_chunks import load_chunks
from training_loop.dataset.split_chunks import split_chunks
from training_loop.dataset.ChunkDataset import ChunkDataset
from training_loop.dataset.csv_logger import CSVLogger
from training_loop.evaluate_on_chunks import evaluate_on_chunks


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_transformer():
    """Train the fixed Transformer Cholesky model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and split dataset
    chunks = load_chunks("dataset_smaller")
    train_chunks, val_chunks, test_chunks = split_chunks(chunks)

    print(f"Train chunks: {len(train_chunks)}")
    print(f"Val chunks: {len(val_chunks)}")
    print(f"Test chunks: {len(test_chunks)}")

    # Create output directories
    ckpt_dir = "checkpoints_3/transformer_cholesky_v2"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs("csvs_3", exist_ok=True)

    # CSV logger
    csv_logger = CSVLogger(csv_dir="csvs_3", name="transformer_cholesky_v2")

    # Model + optimizer
    model = TransformerCholeskyAutoencoder(loss_fn=FrobeniusFidelityLoss()).to(device)

    # Hyperparameters
    lr = 3e-4
    weight_decay = 1e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    EPOCHS = 100
    EARLY_STOP_PATIENCE = 15
    BATCH = 8  # Transformer needs smaller batches

    n_params = count_parameters(model)

    print(f"\n{'='*60}")
    print(f"Training transformer_cholesky_v2 (FIXED architecture)")
    print(f"{'='*60}")
    print(f"Parameters: {n_params:,}")
    print(f"Batch size: {BATCH} | LR: {lr} | Weight decay: {weight_decay}")
    print(f"Early stopping patience: {EARLY_STOP_PATIENCE}")
    print(f"Output: Cholesky-constrained density matrix")
    print(f"")
    print(f"Fix applied:")
    print(f"  - CLS token used for global prediction")
    print(f"  - Learnable decoder query (no degenerate self-attention)")
    print(f"  - Global projection: CLS -> 1024 Cholesky params")
    print(f"{'='*60}\n")

    best_val = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}")

        # Training over all chunks
        model.train()
        for c_idx, (X, Y) in enumerate(train_chunks):
            ds = ChunkDataset(X, Y, "transformer")
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
        val_loss = evaluate_on_chunks(model, val_chunks, "transformer", device, BATCH)
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

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Best validation loss: {best_val:.6f}")
    print(f"Checkpoints saved to: {ckpt_dir}/")
    print(f"Training logs saved to: csvs_3/transformer_cholesky_v2.csv")
    print(f"{'='*60}")


if __name__ == "__main__":
    train_transformer()
