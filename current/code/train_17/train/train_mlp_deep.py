"""
Train Deep MLP on 5-qubit dataset WITH FROBENIUS NORMALIZATION.

Usage:
    python train_17/train/train_mlp_deep.py
"""

import os
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import torch
from torch.utils.data import Dataset, DataLoader

from train_16.models.mlp_5qubit_deep import HierarchicalMLP5QubitDeep
from losses.frob import FrobeniusFidelityLoss

from training_loop.dataset.load_chunks import load_chunks
from training_loop.dataset.split_chunks import split_chunks
from training_loop.dataset.csv_logger import CSVLogger


def normalize_to_unit(x):
    """Normalize tensor to unit Frobenius norm."""
    norm = x.flatten().norm() + 1e-8
    return x / norm


class FrobeniusNormalizedDataset(Dataset):
    """Dataset that normalizes density matrices to unit Frobenius norm."""

    def __init__(self, X, Y, mode):
        self.X = X.double()
        self.Y = Y.double()
        self.mode = mode

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].permute(2, 0, 1)
        y = self.Y[idx].permute(2, 0, 1)
        return normalize_to_unit(x), normalize_to_unit(y)


def evaluate_on_chunks(model, chunks, device, batch_size):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for X, Y in chunks:
            ds = FrobeniusNormalizedDataset(X, Y, "mlp")
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

            for x, y in loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = model.compute_loss(pred, y)
                total_loss += loss.item() * x.shape[0]
                total_samples += x.shape[0]

    return total_loss / total_samples


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    chunks = load_chunks("dataset_5qubit_float64")
    train_chunks, val_chunks, test_chunks = split_chunks(chunks)

    print(f"Train chunks: {len(train_chunks)}")
    print(f"Val chunks: {len(val_chunks)}")
    print(f"Test chunks: {len(test_chunks)}")
    print(f"\nUsing FROBENIUS NORMALIZATION")

    ckpt_dir = os.path.join(base_dir, "checkpoints_17", "mlp_deep")
    os.makedirs(ckpt_dir, exist_ok=True)
    csv_logger = CSVLogger(csv_dir=os.path.join(base_dir, "csvs_17"), name="mlp_deep")

    model = HierarchicalMLP5QubitDeep(loss_fn=FrobeniusFidelityLoss()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

    EPOCHS = 100
    EARLY_STOP_PATIENCE = 15
    BATCH = 64

    print(f"\n{'=' * 60}")
    print(f"Training MLP Deep (v17 - Frobenius normalized)")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Batch size: {BATCH}")
    print(f"{'=' * 60}\n")

    best_val = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}")
        model.train()
        train_loss_val = 0.0

        for c_idx, (X, Y) in enumerate(train_chunks):
            ds = FrobeniusNormalizedDataset(X, Y, "mlp")
            loader = DataLoader(ds, batch_size=BATCH, shuffle=True)

            for b_idx, (x, y) in enumerate(loader):
                x, y = x.to(device), y.to(device)
                pred = model(x)
                train_loss = model.compute_loss(pred, y)
                train_loss_val = train_loss.item()

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                csv_logger.log_train(epoch, c_idx, b_idx, train_loss_val)

            print(f"  chunk {c_idx} final batch loss = {train_loss_val:.6f}")

        val_loss = evaluate_on_chunks(model, val_chunks, device, BATCH)
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
