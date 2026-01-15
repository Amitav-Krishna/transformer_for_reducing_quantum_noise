"""
Train Residual MLP and Transformer on 5-qubit dataset WITH FROBENIUS NORMALIZATION.

v19 key change: Seed ablation with FIXED data split (seed=42).
The --seed argument now controls model initialization, not data split.

Checkpoints saved to: train_19/checkpoints_19_seed{SEED}/
Training logs saved to: train_19/csvs_19_seed{SEED}/

Usage:
    python train_19/train.py --seed 100 --model transformer
    python train_19/train.py --seed 100 --model mlp
    python train_19/train.py --seed 200 --model transformer
    python train_19/train.py --seed 200 --model mlp
"""

import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import torch
from torch.utils.data import Dataset, DataLoader

# Reuse components from train_18
from train_18.models.mlp_5qubit import HierarchicalMLP5Qubit
from train_18.models.transformer_5qubit import HierarchicalTransformer5Qubit
from train_18.losses.frob import FrobeniusFidelityLoss
from train_18.dataset.load_chunks import load_chunks
from train_18.dataset.split_chunks import split_chunks
from train_18.dataset.csv_logger import CSVLogger


def normalize_to_unit(x):
    """
    Normalize tensor to unit Frobenius norm.

    Args:
        x: (2, H, W) tensor

    Returns:
        Normalized tensor with Frobenius norm = 1
    """
    norm = x.flatten().norm() + 1e-8
    return x / norm


class FrobeniusNormalizedDataset(Dataset):
    """
    Dataset that normalizes density matrices to unit Frobenius norm.

    This ensures both noisy input and clean target have ||rho||_F = 1.

    Input format: (H, W, 2) where channel 0 = real, channel 1 = imag
    Output format: (2, H, W) for model input
    """

    def __init__(self, X, Y, mode):
        self.X = X.double()
        self.Y = Y.double()
        self.mode = mode

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]  # (H, W, 2)
        y = self.Y[idx]  # (H, W, 2)

        # Permute to (2, H, W) for model
        x = x.permute(2, 0, 1)
        y = y.permute(2, 0, 1)

        # Normalize to unit Frobenius norm
        x = normalize_to_unit(x)
        y = normalize_to_unit(y)

        return x, y


def evaluate_on_chunks(model, chunks, arch, device, batch_size):
    """Evaluate model on validation/test chunks."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for X, Y in chunks:
            ds = FrobeniusNormalizedDataset(X, Y, arch)
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


def train_single_experiment(name, model, arch, train_chunks, val_chunks, device, base_dir, seed):
    """Train a single model."""
    # Directories with seed in name
    ckpt_dir = os.path.join(base_dir, f"checkpoints_19_seed{seed}", name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # CSV logger with seed in name
    csv_dir = os.path.join(base_dir, f"csvs_19_seed{seed}")
    os.makedirs(csv_dir, exist_ok=True)
    csv_logger = CSVLogger(csv_dir=csv_dir, name=name)

    # Optimizer
    lr = 3e-4
    weight_decay = 1e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Hyperparameters
    EPOCHS = 100
    EARLY_STOP_PATIENCE = 15
    if arch == "mlp":
        BATCH = 64
    else:
        BATCH = 8  # Transformer with 1024 tokens needs smaller batches

    n_params = count_parameters(model)

    print(f"\n{'=' * 60}")
    print(f"Training {name} (v19 - seed ablation, fixed data split)")
    print(f"Model init seed: {seed}")
    print(f"Data split seed: 42 (fixed)")
    print(f"Architecture: {arch}")
    print(f"Parameters: {n_params:,}")
    print(f"Batch size: {BATCH} | LR: {lr} | Weight decay: {weight_decay}")
    print(f"Early stopping patience: {EARLY_STOP_PATIENCE}")
    print(f"{'=' * 60}\n")

    best_val = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}")

        # Training over all chunks
        model.train()
        train_loss_val = 0.0
        for c_idx, (X, Y) in enumerate(train_chunks):
            ds = FrobeniusNormalizedDataset(X, Y, arch)
            loader = DataLoader(ds, batch_size=BATCH, shuffle=True)

            for b_idx, (x, y) in enumerate(loader):
                x, y = x.to(device), y.to(device)

                pred = model(x)
                train_loss = model.compute_loss(pred, y)
                train_loss_val = train_loss.item()

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                csv_logger.log_train(epoch, c_idx, b_idx, float(train_loss_val))

            print(f"  chunk {c_idx} final batch loss = {train_loss_val:.6f}")

        # Validation
        val_loss = evaluate_on_chunks(model, val_chunks, arch, device, BATCH)
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

    print(f"\nTraining complete for {name} with init seed {seed}")
    print(f"Best validation loss: {best_val:.6f}")
    return best_val


def main():
    parser = argparse.ArgumentParser(description="Train 5-qubit models with seed ablation (fixed data split)")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for model initialization")
    parser.add_argument("--model", type=str, required=True, choices=["transformer", "mlp"],
                        help="Model to train: 'transformer' or 'mlp'")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model init seed: {args.seed}")
    print(f"Data split seed: 42 (fixed)")
    print(f"Model: {args.model}")

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Get base directory (train_19/)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load and split dataset with FIXED seed=42
    chunks = load_chunks("dataset_5qubit_float64")
    train_chunks, val_chunks, test_chunks = split_chunks(chunks, seed=42)

    print(f"Train chunks: {len(train_chunks)}")
    print(f"Val chunks: {len(val_chunks)}")
    print(f"Test chunks: {len(test_chunks)}")
    print(f"\nUsing FROBENIUS NORMALIZATION for scale consistency")
    print(f"Data split: seed=42 (fixed)")
    print(f"Model init: seed={args.seed}")

    # Create model (will use the seeded random state)
    if args.model == "transformer":
        model = HierarchicalTransformer5Qubit(loss_fn=FrobeniusFidelityLoss()).to(device)
        arch = "transformer"
    else:
        model = HierarchicalMLP5Qubit(loss_fn=FrobeniusFidelityLoss()).to(device)
        arch = "mlp"

    # Train
    train_single_experiment(
        args.model, model, arch, train_chunks, val_chunks, device, base_dir, args.seed
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Checkpoints saved to: {os.path.join(base_dir, f'checkpoints_19_seed{args.seed}/')}")
    print(f"Training logs saved to: {os.path.join(base_dir, f'csvs_19_seed{args.seed}/')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
