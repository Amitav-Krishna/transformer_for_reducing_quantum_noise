"""
Train MLP baseline for density matrix denoising.

This trains an MLP with ~123k parameters (matching the small transformer)
to serve as a baseline that also has a global receptive field.

Usage:
    python train_mlp.py
"""

import torch

from models.mlp import MLPAutoencoder
from losses.frob import FrobeniusFidelityLoss
from training_loop.dataset.load_chunks import load_chunks
from training_loop.dataset.split_chunks import split_chunks
from training_loop.train_single_experiment import train_single_experiment


EXPERIMENTS = {
    "mlp_frob": {
        "arch": "mlp",
        "loss": "frob",
        "create_model": lambda: MLPAutoencoder(loss_fn=FrobeniusFidelityLoss()),
    },
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and split dataset
    chunks = load_chunks("dataset_smaller")
    train_chunks, val_chunks, test_chunks = split_chunks(chunks)

    print(f"Train chunks: {len(train_chunks)}")
    print(f"Val chunks: {len(val_chunks)}")
    print(f"Test chunks: {len(test_chunks)}")

    # Train MLP
    for name, config in EXPERIMENTS.items():
        train_single_experiment(name, config, train_chunks, val_chunks, device)

    print("\nTraining complete!")
    print("Checkpoint saved to: checkpoints_2/mlp_frob/best.pt")
    print("Training log saved to: csvs_2/mlp_frob.csv")


if __name__ == "__main__":
    main()