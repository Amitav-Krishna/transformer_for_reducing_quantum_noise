"""
Train the properly-designed capacity-matched transformer (v3).

V3 fixes the issues with v1 and v2:
1. Proper FFN ratio: ffn_dim = 4 * embed_dim (standard transformer design)
2. Proper bottleneck: 50% compression
3. Lower learning rate for larger model

Run this script to train both Frobenius and Physics variants:
    python train_transformer_matched_v3.py

Checkpoints saved to: checkpoints_2/transformer_matched_v3_{frob,physics}/
Training logs saved to: csvs_2/transformer_matched_v3_{frob,physics}.csv
"""

import os
import torch

from models.transformer_matched_v3 import TransformerAutoencoderMatchedV3
from losses.frob import FrobeniusFidelityLoss
from losses.total_physics_loss import CompositePhysicsTotalLoss
from training_loop.dataset.load_chunks import load_chunks
from training_loop.dataset.split_chunks import split_chunks
from training_loop.train_single_experiment import train_single_experiment
from training_loop.evaluate_all_on_test import evaluate_all_checkpoints_on_test


EXPERIMENTS = {
    "transformer_matched_v3_frob": {
        "arch": "transformer",
        "loss": "frob",
        "lr": 1e-4,  # Lower LR for larger model
        "create_model": lambda: TransformerAutoencoderMatchedV3(loss_fn=FrobeniusFidelityLoss())
    },
    "transformer_matched_v3_physics": {
        "arch": "transformer",
        "loss": "physics",
        "lr": 1e-4,  # Lower LR for larger model
        "create_model": lambda: TransformerAutoencoderMatchedV3(loss_fn=CompositePhysicsTotalLoss())
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

    print("\nDone! Now run the following to get evaluations:")
    print("  python eval_transformer_matched_v3_uhlmann.py")
    print("  python eval_transformer_matched_v3_per_noise_cell.py")


if __name__ == "__main__":
    main()