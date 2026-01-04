"""
Train Hierarchical Transformer on 6-qubit dataset.

This demonstrates that the Transformer approach scales beyond 5 qubits
with hierarchical tokenization (64 tokens instead of 4096).

Checkpoints saved to: train_14/checkpoints_14/
Training logs saved to: train_14/csvs_14/

Usage:
    python train_14/train.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, Dataset
import glob

from train_14.transformer_6qubit import HierarchicalTransformer6Qubit, count_parameters
from losses.frob import FrobeniusFidelityLoss


class Dataset6Qubit(Dataset):
    """Load 6-qubit dataset from chunks."""

    def __init__(self, chunk_files):
        self.X_list = []
        self.Y_list = []

        for f in chunk_files:
            data = torch.load(f)
            # Data is (N, 64, 64, 2), need (N, 2, 64, 64)
            X = data["X"].permute(0, 3, 1, 2)
            Y = data["Y"].permute(0, 3, 1, 2)
            self.X_list.append(X)
            self.Y_list.append(Y)

        self.X = torch.cat(self.X_list, dim=0)
        self.Y = torch.cat(self.Y_list, dim=0)
        print(f"Loaded {len(self.X)} samples from {len(chunk_files)} chunks")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def load_6qubit_dataset(data_dir="dataset_6qubit", train_ratio=0.8, val_ratio=0.1):
    """Load and split 6-qubit dataset."""
    chunk_files = sorted(glob.glob(f"{data_dir}/chunk_*.pt"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found in {data_dir}")

    print(f"Found {len(chunk_files)} chunk files")

    # Split chunks 80/10/10
    n = len(chunk_files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_chunks = chunk_files[:n_train]
    val_chunks = chunk_files[n_train:n_train + n_val]
    test_chunks = chunk_files[n_train + n_val:]

    print(f"Train chunks: {len(train_chunks)}")
    print(f"Val chunks: {len(val_chunks)}")
    print(f"Test chunks: {len(test_chunks)}")

    return train_chunks, val_chunks, test_chunks


class CSVLogger:
    """Simple CSV logger for training metrics."""

    def __init__(self, csv_dir, name):
        os.makedirs(csv_dir, exist_ok=True)
        self.train_file = open(f"{csv_dir}/{name}_train.csv", "w")
        self.val_file = open(f"{csv_dir}/{name}_val.csv", "w")
        self.train_file.write("epoch,batch,loss\n")
        self.val_file.write("epoch,loss\n")

    def log_train(self, epoch, batch, loss):
        self.train_file.write(f"{epoch},{batch},{loss}\n")
        # Don't flush on every batch to avoid disk quota issues

    def log_val(self, epoch, loss):
        self.val_file.write(f"{epoch},{loss}\n")
        try:
            self.val_file.flush()
            self.train_file.flush()  # Flush train file at end of epoch
        except OSError:
            pass  # Ignore disk quota errors

    def close(self):
        self.train_file.close()
        self.val_file.close()


def evaluate(model, val_loader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = model.compute_loss(pred, y)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Directories
    ckpt_dir = os.path.join(base_dir, "checkpoints_14", "hierarchical_transformer")
    csv_dir = os.path.join(base_dir, "csvs_14")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    # Load dataset
    train_chunks, val_chunks, test_chunks = load_6qubit_dataset("dataset_6qubit")

    train_dataset = Dataset6Qubit(train_chunks)
    val_dataset = Dataset6Qubit(val_chunks)

    # Delete dataset from disk to free up space (data is now in RAM)
    import shutil
    print("Deleting dataset from disk to free space...")
    shutil.rmtree("dataset_6qubit", ignore_errors=True)
    print("Dataset deleted. Data is loaded in RAM.")

    # Create data loaders
    BATCH = 16  # Smaller batches for 6-qubit (larger matrices)
    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False, num_workers=0)

    # Model
    model = HierarchicalTransformer6Qubit(loss_fn=FrobeniusFidelityLoss()).to(device)
    n_params = count_parameters(model)

    # Optimizer
    lr = 3e-4
    weight_decay = 1e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Hyperparameters
    EPOCHS = 100
    EARLY_STOP_PATIENCE = 15

    print(f"\n{'='*60}")
    print(f"Training Hierarchical Transformer (6-qubit)")
    print(f"Parameters: {n_params:,}")
    print(f"Batch size: {BATCH} | LR: {lr} | Weight decay: {weight_decay}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Early stopping patience: {EARLY_STOP_PATIENCE}")
    print(f"{'='*60}\n")

    # CSV logger
    csv_logger = CSVLogger(csv_dir, "hierarchical_transformer")

    best_val = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}")

        # Training
        model.train()
        epoch_loss = 0
        n_batches = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = model.compute_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            csv_logger.log_train(epoch, batch_idx, loss.item())

            if (batch_idx + 1) % 100 == 0:
                print(f"  batch {batch_idx + 1}/{len(train_loader)}, loss = {loss.item():.6f}")

        avg_train_loss = epoch_loss / n_batches
        print(f"  Train loss: {avg_train_loss:.6f}")

        # Validation
        val_loss = evaluate(model, val_loader, device)
        print(f"  Val loss: {val_loss:.6f}")
        csv_logger.log_val(epoch, val_loss)

        # Save best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            epochs_without_improvement = 0
            best_path = os.path.join(ckpt_dir, "best.pt")
            try:
                torch.save(model.state_dict(), best_path)
                print(f"  Saved BEST checkpoint -> {best_path}")
            except OSError as e:
                print(f"  Warning: Could not save checkpoint: {e}")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epochs")

        # Skip per-epoch checkpoints to save disk space
        # epoch_path = os.path.join(ckpt_dir, f"epoch_{epoch}.pt")
        # torch.save(model.state_dict(), epoch_path)

        # Early stopping
        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            print(f"Best validation loss: {best_val:.6f}")
            break

    csv_logger.close()
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Checkpoint: {ckpt_dir}")
    print(f"Logs: {csv_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
