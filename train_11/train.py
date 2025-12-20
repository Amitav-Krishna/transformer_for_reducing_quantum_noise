"""
Train MLP and Transformer on Pauli basis representation.

v11 tests whether the Transformer's advantage persists when using a different
input representation (Pauli basis instead of flattened real/imag).

Hypothesis: Pauli basis representation is more "flat" (no spatial structure),
so MLPs should perform better relative to Transformers. If Transformer still
wins significantly, it proves the advantage is fundamental, not just due to
the flattened representation suiting attention.

Usage:
    python train_11/train.py mlp      # Train MLP on Pauli basis
    python train_11/train.py transformer  # Train Transformer on Pauli basis
    python train_11/train.py both     # Train both
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, TensorDataset

from train_11.pauli_representation import (
    density_matrix_to_pauli_basis,
    PauliRepresentationDataset
)
from train_11.mlp_pauli import MLPPauliAutoencoder
from train_11.transformer_pauli import TransformerPauliAutoencoder

from losses.frob import FrobeniusFidelityLoss
from training_loop.dataset.load_chunks import load_chunks
from training_loop.dataset.split_chunks import split_chunks
from training_loop.dataset.csv_logger import CSVLogger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def convert_chunks_to_pauli(chunks, device):
    """Convert data chunks to Pauli basis representation."""
    pauli_chunks = []
    for X, Y in chunks:
        # X, Y are (N, 2, 32, 32)
        X_pauli = torch.tensor([
            density_matrix_to_pauli_basis(X[i], n_qubits=5)
            for i in range(len(X))
        ], dtype=torch.float32, device=device)

        Y_pauli = torch.tensor([
            density_matrix_to_pauli_basis(Y[i], n_qubits=5)
            for i in range(len(Y))
        ], dtype=torch.float32, device=device)

        pauli_chunks.append((X_pauli, Y_pauli))

    return pauli_chunks


def train_model(model_type, train_chunks, val_chunks, device):
    """Train MLP or Transformer on Pauli basis."""

    if model_type == "mlp":
        name = "mlp_pauli_frob"
        model = MLPPauliAutoencoder(loss_fn=FrobeniusFidelityLoss()).to(device)
    else:  # transformer
        name = "transformer_pauli_frob"
        model = TransformerPauliAutoencoder(loss_fn=FrobeniusFidelityLoss()).to(device)

    n_params = count_parameters(model)

    # Directories
    ckpt_dir = f"checkpoints_11/{name}"
    os.makedirs(ckpt_dir, exist_ok=True)

    # CSV logger
    csv_logger = CSVLogger(csv_dir="csvs_11", name=name)

    # Optimizer
    lr = 3e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Hyperparameters
    EPOCHS = 100
    BATCH = 64 if model_type == "mlp" else 8
    PATIENCE = 15
    patience_counter = 0

    print(f"\n{'='*70}")
    print(f"Training {model_type.upper()} on Pauli Basis Representation (v11)")
    print(f"{'='*70}")
    print(f"Input representation: Pauli coefficients (1024-dim)")
    print(f"Parameters: {n_params:,}")
    print(f"Batch size: {BATCH} | LR: {lr} | Patience: {PATIENCE} epochs")
    print(f"Checkpoints: checkpoints_11/{name}/")
    print(f"Logs: csvs_11/{name}.csv")
    print(f"{'='*70}\n")

    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}")

        # Training over all chunks
        model.train()
        for c_idx, (X_pauli, Y_pauli) in enumerate(train_chunks):
            # Create DataLoader
            dataset = TensorDataset(X_pauli, Y_pauli)
            loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)

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
        model.eval()
        val_loss_total = 0.0
        val_count = 0

        with torch.no_grad():
            for X_pauli, Y_pauli in val_chunks:
                dataset = TensorDataset(X_pauli, Y_pauli)
                loader = DataLoader(dataset, batch_size=BATCH, shuffle=False)

                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    batch_loss = model.compute_loss(pred, y)
                    val_loss_total += float(batch_loss.item()) * len(x)
                    val_count += len(x)

        val_loss = val_loss_total / val_count
        print(f"Validation loss: {val_loss:.6f}")
        csv_logger.log_val(epoch, val_loss)

        # Save best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(ckpt_dir, "best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"✓ Best validation loss: {best_val:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{PATIENCE})")

        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"Best checkpoint: {ckpt_dir}/best.pt")
    print(f"Training logs: csvs_11/{name}.csv")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        choices=["mlp", "transformer", "both"],
        help="Which model to train"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and split dataset
    print("Loading dataset...")
    chunks = load_chunks("dataset_smaller")
    train_chunks, val_chunks, test_chunks = split_chunks(chunks)

    print(f"Train chunks: {len(train_chunks)}")
    print(f"Val chunks: {len(val_chunks)}")
    print(f"Test chunks: {len(test_chunks)}")

    # Convert to Pauli basis
    print("\nConverting to Pauli basis representation...")
    train_chunks_pauli = convert_chunks_to_pauli(train_chunks, device)
    val_chunks_pauli = convert_chunks_to_pauli(val_chunks, device)

    # Train requested model(s)
    if args.model in ["mlp", "both"]:
        train_model("mlp", train_chunks_pauli, val_chunks_pauli, device)

    if args.model in ["transformer", "both"]:
        train_model("transformer", train_chunks_pauli, val_chunks_pauli, device)


if __name__ == "__main__":
    main()
