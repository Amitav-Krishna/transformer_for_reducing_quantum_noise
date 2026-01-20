"""
Evaluate Hierarchical MLP models on Uhlmann fidelity.

Supports 5-qubit and 8-qubit models. Model and checkpoint are auto-detected
based on dataset directory.

Usage:
    python train_15/eval_uhlmann.py --model 5qubit
    python train_15/eval_uhlmann.py --model 5qubit_wide
    python train_15/eval_uhlmann.py --model 8qubit

Outputs:
- csvs_15/hierarchical_mlp_{model}_test_uhlmann.csv
"""

import os
import sys
import csv
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from losses.frob import FrobeniusFidelityLoss
from training_loop.dataset.load_chunks import load_chunks
from training_loop.dataset.split_chunks import split_chunks
from training_loop.dataset.ChunkDataset import ChunkDataset


def matrix_sqrt(A):
    """
    Compute matrix square root via eigendecomposition.
    A must be Hermitian positive semidefinite.
    """
    eigvals, eigvecs = torch.linalg.eigh(A)
    eigvals = torch.clamp(eigvals, min=0)  # numerical stability
    sqrt_eigvals = torch.sqrt(eigvals)
    sqrt_diag = torch.diag_embed(sqrt_eigvals.to(eigvecs.dtype))
    return eigvecs @ sqrt_diag @ eigvecs.mH


def uhlmann_fidelity_batch(rho_pred, rho_true, eps=1e-10):
    """
    Compute Uhlmann fidelity between predicted and true density matrices.

    F(rho, sigma) = (Tr[sqrt(sqrt(rho) sigma sqrt(rho))])^2

    rho_pred, rho_true: (B, 2, N, N) with [real, imag] channels
    Returns: (B,) tensor of fidelities in [0, 1]
    """
    # Convert to complex matrices (B, N, N)
    rho = rho_pred[:, 0] + 1j * rho_pred[:, 1]
    sigma = rho_true[:, 0] + 1j * rho_true[:, 1]

    B = rho.shape[0]
    fidelities = []

    for i in range(B):
        r = rho[i]
        s = sigma[i]

        # Ensure Hermitian (average with conjugate transpose)
        r = (r + r.mH) / 2
        s = (s + s.mH) / 2

        # Compute sqrt(rho)
        sqrt_r = matrix_sqrt(r)

        # Compute sqrt(rho) sigma sqrt(rho)
        inner = sqrt_r @ s @ sqrt_r

        # Ensure Hermitian for numerical stability
        inner = (inner + inner.mH) / 2

        # Compute sqrt(sqrt(rho) sigma sqrt(rho))
        sqrt_inner = matrix_sqrt(inner)

        # F = (Tr[sqrt(sqrt(rho) sigma sqrt(rho))])^2
        trace_val = torch.trace(sqrt_inner).real
        fid = trace_val**2

        # Clamp to valid range
        fid = torch.clamp(fid, 0.0, 1.0)
        fidelities.append(fid)

    return torch.stack(fidelities)


@torch.no_grad()
def evaluate_uhlmann_on_chunks(model, chunks, device, batch_size=8):
    """
    Compute mean Uhlmann fidelity over all chunks.
    Returns: (mean_fidelity, std_fidelity, all_fidelities)
    """
    model.eval()
    all_fid = []

    for X, Y in chunks:
        ds = ChunkDataset(X, Y, mode="transformer")
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            fid = uhlmann_fidelity_batch(pred, y)
            all_fid.append(fid.cpu())

    all_fid = torch.cat(all_fid)
    return all_fid.mean().item(), all_fid.std().item(), all_fid


@torch.no_grad()
def evaluate_baseline_uhlmann(chunks, device, batch_size=8):
    """
    Compute baseline Uhlmann fidelity (noisy vs clean, no model).
    """
    all_fid = []

    for X, Y in chunks:
        ds = ChunkDataset(X, Y, mode="transformer")
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            # Baseline: compare noisy input directly to clean target
            fid = uhlmann_fidelity_batch(x, y)
            all_fid.append(fid.cpu())

    all_fid = torch.cat(all_fid)
    return all_fid.mean().item(), all_fid.std().item()


def get_model_class(model_name: str):
    """Get model class and associated settings."""
    if model_name == "5qubit":
        from train_15.mlp_hierarchical_5qubit import HierarchicalMLP5Qubit

        return HierarchicalMLP5Qubit, "5qubit", 32
    elif model_name == "5qubit_wide":
        from train_15.mlp_hierarchical_5qubit_wide import HierarchicalMLP5QubitWide

        return HierarchicalMLP5QubitWide, "5qubit", 32
    elif model_name == "8qubit":
        from train_15.mlp_hierarchical_8qubit import HierarchicalMLP8Qubit

        return HierarchicalMLP8Qubit, "8qubit", 256
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate hierarchical MLP on Uhlmann fidelity"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["5qubit", "5qubit_wide", "8qubit"],
        help="Model to evaluate",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get model class and settings
    ModelClass, qubit_size, matrix_size = get_model_class(args.model)

    # Determine paths based on qubit size and environment
    if qubit_size == "5qubit":
        if os.path.exists("/workspace/dataset_smaller"):
            dataset_dir = "/workspace/dataset_smaller"
            checkpoint_dir = "/workspace/train_15/checkpoints_15"
            csv_dir = "/workspace/csvs_15"
        else:
            dataset_dir = "dataset_smaller"
            checkpoint_dir = "train_15/checkpoints_15"
            csv_dir = "csvs_15"
    else:  # 8qubit
        if os.path.exists("/workspace/dataset_8qubit"):
            dataset_dir = "/workspace/dataset_8qubit"
            checkpoint_dir = "/workspace/train_15/checkpoints_15"
            csv_dir = "/workspace/csvs_15"
        else:
            dataset_dir = "dataset_8qubit"
            checkpoint_dir = "train_15/checkpoints_15"
            csv_dir = "csvs_15"

    checkpoint_path = os.path.join(
        checkpoint_dir, f"hierarchical_mlp_{args.model}", "best.pt"
    )
    os.makedirs(csv_dir, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {dataset_dir}...")
    chunks = load_chunks(dataset_dir)
    if len(chunks) == 0:
        print(f"ERROR: No chunks found in {dataset_dir}")
        sys.exit(1)

    _, _, test_chunks = split_chunks(chunks, train_ratio=0.8, val_ratio=0.1, seed=42)
    print(f"Test chunks: {len(test_chunks)}")
    test_samples = sum(X.shape[0] for X, Y in test_chunks)
    print(f"Test samples: {test_samples:,}")

    # Load model
    print(f"\nLoading model from {checkpoint_path}...")
    model = ModelClass(loss_fn=FrobeniusFidelityLoss()).to(device)

    try:
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        print("Make sure training has completed first.")
        sys.exit(1)

    # Evaluate baseline
    print("\nComputing baseline Uhlmann fidelity (noisy vs clean)...")
    baseline_mean, baseline_std = evaluate_baseline_uhlmann(test_chunks, device)
    print(f"Baseline Uhlmann Fidelity: {baseline_mean:.4f} +/- {baseline_std:.4f}")

    # Evaluate model
    print("\nComputing model Uhlmann fidelity...")
    model_mean, model_std, all_fid = evaluate_uhlmann_on_chunks(
        model, test_chunks, device
    )
    print(f"Model Uhlmann Fidelity: {model_mean:.4f} +/- {model_std:.4f}")

    # Compute improvement
    improvement = model_mean / baseline_mean if baseline_mean > 0 else float("inf")
    print(f"\nImprovement over baseline: {improvement:.2f}x")

    # Save results
    results_path = os.path.join(
        csv_dir, f"hierarchical_mlp_{args.model}_test_uhlmann.csv"
    )
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["baseline_mean", baseline_mean])
        writer.writerow(["baseline_std", baseline_std])
        writer.writerow(["model_mean", model_mean])
        writer.writerow(["model_std", model_std])
        writer.writerow(["improvement_ratio", improvement])
        writer.writerow(["test_samples", test_samples])

    print(f"\nResults saved to: {results_path}")

    # Summary
    print(f"\n{'=' * 50}")
    print(f"SUMMARY: Hierarchical MLP ({args.model})")
    print(f"{'=' * 50}")
    print(f"{'Metric':<25} {'Value':>15}")
    print("-" * 50)
    print(f"{'Baseline (noisy)':<25} {baseline_mean:>15.4f}")
    print(f"{'Model (MLP)':<25} {model_mean:>15.4f}")
    print(f"{'Improvement':<25} {improvement:>15.2f}x")


if __name__ == "__main__":
    main()
