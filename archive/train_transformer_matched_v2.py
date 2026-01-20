"""
Train the fixed capacity-matched transformer (v2) with corrected bottleneck.

Run this script to train both Frobenius and Physics variants:
    python train_transformer_matched_v2.py

Checkpoints saved to: checkpoints_2/transformer_matched_v2_{frob,physics}/
Training logs saved to: csvs_2/transformer_matched_v2_{frob,physics}.csv
"""

import os
import torch

from models.transformer_matched_v2 import TransformerAutoencoderMatchedV2
from losses.frob import FrobeniusFidelityLoss
from losses.total_physics_loss import CompositePhysicsTotalLoss
from training_loop.dataset.load_chunks import load_chunks
from training_loop.dataset.split_chunks import split_chunks
from training_loop.train_single_experiment import train_single_experiment
from training_loop.evaluate_all_on_test import evaluate_all_checkpoints_on_test


EXPERIMENTS = {
    "transformer_matched_v2_frob": {
        "arch": "transformer",
        "loss": "frob",
        "create_model": lambda: TransformerAutoencoderMatchedV2(loss_fn=FrobeniusFidelityLoss())
    },
    "transformer_matched_v2_physics": {
        "arch": "transformer",
        "loss": "physics",
        "create_model": lambda: TransformerAutoencoderMatchedV2(loss_fn=CompositePhysicsTotalLoss())
    },
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and split dataset
    chunks = load_chunks("dataset_smaller")
    train_chunks, val_chunks, test_chunks = split_chunks(chunks)

    # Train both variants
    for name, config in EXPERIMENTS.items():
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"{'='*60}")
        train_single_experiment(name, config, train_chunks, val_chunks, device)

    # Evaluate on test set
    print(f"\n{'='*60}")
    print("Evaluating on test set...")
    print(f"{'='*60}")
    test_results = evaluate_all_checkpoints_on_test(EXPERIMENTS, test_chunks, device)

    print("\nDone! Now run the following to get per-noise-cell evaluations:")
    print("  python eval_per_noise_cell.py")
    print("  python eval_models_on_uhlmann.py")


if __name__ == "__main__":
    main()