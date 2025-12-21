"""
Train MLP and Transformer on Pauli basis representation.

v11 tests whether the Transformer's advantage persists when using a different
input representation (Pauli basis instead of flattened real/imag).

Hypothesis: Pauli basis representation is more "flat" (no spatial structure),
so MLPs should perform better relative to Transformers. If Transformer still
wins significantly, it proves the advantage is fundamental, not just due to
the flattened representation suiting attention.

Usage:
    python train_11/train.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, TensorDataset

from train_11.pauli_representation import density_matrix_to_pauli_basis
from train_11.mlp_pauli import MLPPauliAutoencoder
from train_11.transformer_pauli import TransformerPauliAutoencoder
from train_11.pauli_loss import PauliFrobeniusLoss
from training_loop.dataset.load_chunks import load_chunks
from training_loop.dataset.split_chunks import split_chunks
from training_loop.dataset.csv_logger import CSVLogger


EXPERIMENTS = {
    "mlp_pauli_frob": {
        "arch": "mlp",
        "loss": "frob",
        "create_model": lambda: MLPPauliAutoencoder(loss_fn=PauliFrobeniusLoss()),
        "batch_size": 64,
    },
    "transformer_pauli_frob": {
        "arch": "transformer",
        "loss": "frob",
        "create_model": lambda: TransformerPauliAutoencoder(loss_fn=PauliFrobeniusLoss()),
        "batch_size": 8,
    },
}


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


def train_single_experiment(name, config, train_chunks, val_chunks, device):
    """Train a single model on Pauli basis representation."""

    arch = config["arch"]
    loss = config["loss"]
    batch_size = config["batch_size"]

    # Directories
    ckpt_dir = f"checkpoints_11/{name}"
    os.makedirs(ckpt_dir, exist_ok=True)

    # CSV logger
    csv_logger = CSVLogger(csv_dir="csvs_11", name=name)

    # Model + optimizer
    model = config["create_model"]().to(device)
    n_params = count_parameters(model)
    lr = 3e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Hyperparameters
    EPOCHS = 100
    PATIENCE = 15
    patience_counter = 0

    print(f"\n{'='*70}")
    print(f"Training {name}")
    print(f"Architecture: {arch} | Loss: {loss}")
    print(f"Input representation: Pauli basis (1024-dim coefficients)")
    print(f"Parameters: {n_params:,}")
    print(f"Batch size: {batch_size} | LR: {lr} | Patience: {PATIENCE} epochs")
    print(f"{'='*70}\n")

    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}")

        # Training over all chunks
        model.train()
        for c_idx, (X_pauli, Y_pauli) in enumerate(train_chunks):
            dataset = TensorDataset(X_pauli, Y_pauli)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

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

    print(f"\nTraining complete!")
    print(f"Checkpoint saved to: checkpoints_11/{name}/best.pt")
    print(f"Training log saved to: csvs_11/{name}.csv")


def main():
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
    print("\nConverting dataset to Pauli basis representation...")
    train_chunks_pauli = convert_chunks_to_pauli(train_chunks, device)
    val_chunks_pauli = convert_chunks_to_pauli(val_chunks, device)

    # Train all experiments sequentially
    for name, config in EXPERIMENTS.items():
        train_single_experiment(name, config, train_chunks_pauli, val_chunks_pauli, device)

    print("\n" + "="*70)
    print("All training complete!")
    print("="*70)
    print("Checkpoints: checkpoints_11/mlp_pauli_frob/best.pt")
    print("             checkpoints_11/transformer_pauli_frob/best.pt")
    print("Logs: csvs_11/mlp_pauli_frob.csv")
    print("      csvs_11/transformer_pauli_frob.csv")


if __name__ == "__main__":
    main()
