#!/usr/bin/env python3
"""
Uhlmann fidelity evaluation for train_17 models (Frobenius normalized).

Evaluates MLP and Transformer 5-qubit models trained with Frobenius normalization.
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training_loop.dataset.ChunkDataset import ChunkDataset
from training_loop.dataset.load_chunks import load_chunks
from training_loop.dataset.split_chunks import split_chunks

from train_16.models.transformer_5qubit import HierarchicalTransformer5Qubit
from train_16.models.mlp_5qubit import HierarchicalMLP5Qubit

DTYPE = torch.float64
CDTYPE = torch.complex128


def normalize_to_unit(x):
    """Normalize tensor to unit Frobenius norm."""
    norm = x.flatten().norm() + 1e-8
    return x / norm


def matrix_sqrt_float64(A):
    A = A.to(CDTYPE)
    eigvals, eigvecs = torch.linalg.eigh(A)
    if torch.isnan(eigvals).any() or torch.isinf(eigvals).any():
        raise RuntimeError("NaN/Inf in eigenvalues")
    eigvals = torch.clamp(eigvals.real, min=0.0)
    sqrt_eigvals = torch.sqrt(eigvals)
    sqrt_diag = torch.diag(sqrt_eigvals.to(CDTYPE))
    return eigvecs @ sqrt_diag @ eigvecs.mH


def uhlmann_fidelity_single(rho, sigma):
    rho = (rho + rho.mH) / 2
    sigma = (sigma + sigma.mH) / 2
    rho = rho / torch.trace(rho)
    sigma = sigma / torch.trace(sigma)
    sqrt_rho = matrix_sqrt_float64(rho)
    inner = sqrt_rho @ sigma @ sqrt_rho
    inner = (inner + inner.mH) / 2
    sqrt_inner = matrix_sqrt_float64(inner)
    trace_val = torch.trace(sqrt_inner).real
    fid = (trace_val**2).item()
    return max(0.0, min(1.0, fid))


def tensor_to_complex(t):
    return t[0].to(CDTYPE) + 1j * t[1].to(CDTYPE)


@torch.no_grad()
def evaluate_baseline_fidelity(chunks, device, batch_size=32):
    """Compute baseline Uhlmann fidelity: F(noisy, clean)."""
    all_fidelities = []
    n_failures = 0
    n_total = 0

    for X, Y in chunks:
        ds = ChunkDataset(X, Y, mode="transformer")
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            for i in range(x.shape[0]):
                n_total += 1
                noisy = tensor_to_complex(x[i])
                clean = tensor_to_complex(y[i])

                try:
                    fid = uhlmann_fidelity_single(noisy, clean)
                    all_fidelities.append(fid)
                except RuntimeError:
                    n_failures += 1

    fids = torch.tensor(all_fidelities, dtype=DTYPE)
    return {
        "mean": fids.mean().item(),
        "std": fids.std().item(),
        "n_samples": len(all_fidelities),
        "n_failures": n_failures,
    }


@torch.no_grad()
def evaluate_model_fidelity_normalized(model, chunks, device, batch_size=4):
    """
    Compute model Uhlmann fidelity: F(pred, clean).

    IMPORTANT: Normalizes inputs to unit Frobenius norm before feeding to model,
    since train_17 models were trained with normalized inputs.
    """
    model.eval()
    all_fidelities = []
    n_failures = 0
    n_total = 0

    for X, Y in chunks:
        ds = ChunkDataset(X, Y, mode="transformer")
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        for x, y in loader:
            x = x.to(device, dtype=DTYPE)
            y = y.to(device, dtype=DTYPE)

            # Normalize inputs (as done during training)
            x_norm = torch.stack([normalize_to_unit(xi) for xi in x])

            pred = model(x_norm)

            for i in range(pred.shape[0]):
                n_total += 1
                pred_mat = tensor_to_complex(pred[i])
                clean_mat = tensor_to_complex(
                    y[i]
                )  # Use original clean, not normalized

                try:
                    fid = uhlmann_fidelity_single(pred_mat, clean_mat)
                    all_fidelities.append(fid)
                except RuntimeError:
                    n_failures += 1

    fids = torch.tensor(all_fidelities, dtype=DTYPE)
    return {
        "mean": fids.mean().item(),
        "std": fids.std().item(),
        "n_samples": len(all_fidelities),
        "n_failures": n_failures,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("\nLoading dataset...")
    chunks = load_chunks("dataset_5qubit_float64")
    _, _, test_chunks = split_chunks(chunks, 0.8, 0.1, seed=42)
    print(f"Test chunks: {len(test_chunks)}")

    # Baseline
    print("\n" + "=" * 60)
    print("BASELINE: F(noisy, clean)")
    print("=" * 60)
    baseline = evaluate_baseline_fidelity(test_chunks, device)
    print(f"  Mean:     {baseline['mean']:.6f}")
    print(f"  Std:      {baseline['std']:.6f}")
    print(f"  Failures: {baseline['n_failures']}")

    # Evaluate MLP
    print("\n" + "=" * 60)
    print("MLP (train_17, Frobenius normalized)")
    print("=" * 60)
    mlp = HierarchicalMLP5Qubit(dtype=DTYPE).to(device)
    mlp.load_state_dict(
        torch.load(
            "train_17/checkpoints_17/mlp/best.pt",
            map_location=device,
            weights_only=False,
        )
    )
    mlp_result = evaluate_model_fidelity_normalized(mlp, test_chunks, device)
    print(f"  Mean:     {mlp_result['mean']:.6f}")
    print(f"  Std:      {mlp_result['std']:.6f}")
    print(f"  Failures: {mlp_result['n_failures']}")
    print(f"  Improvement: {mlp_result['mean'] / baseline['mean']:.2f}x")

    # Evaluate Transformer
    print("\n" + "=" * 60)
    print("Transformer (train_17, Frobenius normalized)")
    print("=" * 60)
    transformer = HierarchicalTransformer5Qubit(dtype=DTYPE).to(device)
    transformer.load_state_dict(
        torch.load(
            "train_17/checkpoints_17/transformer/best.pt",
            map_location=device,
            weights_only=False,
        )
    )
    transformer_result = evaluate_model_fidelity_normalized(
        transformer, test_chunks, device
    )
    print(f"  Mean:     {transformer_result['mean']:.6f}")
    print(f"  Std:      {transformer_result['std']:.6f}")
    print(f"  Failures: {transformer_result['n_failures']}")
    print(f"  Improvement: {transformer_result['mean'] / baseline['mean']:.2f}x")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<25} {'Uhlmann Fidelity':<20} {'vs Baseline':<15}")
    print("-" * 60)
    print(f"{'Baseline (noisy)':<25} {baseline['mean']:<20.6f} {'1.00x':<15}")
    print(
        f"{'MLP (normalized)':<25} {mlp_result['mean']:<20.6f} {mlp_result['mean'] / baseline['mean']:.2f}x"
    )
    print(
        f"{'Transformer (normalized)':<25} {transformer_result['mean']:<20.6f} {transformer_result['mean'] / baseline['mean']:.2f}x"
    )


if __name__ == "__main__":
    main()
