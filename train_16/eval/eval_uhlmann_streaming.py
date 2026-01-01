#!/usr/bin/env python3
"""
Streaming Uhlmann fidelity evaluation for large datasets (e.g., 8-qubit).

Loads chunks one at a time to avoid OOM errors on datasets that don't fit in RAM.
Uses StreamingChunkDataset from training_loop.

Features:
- Full float64 precision throughout
- Self-check: F(clean, clean) = 1.0 verification
- Skip-and-count failure tracking for eigendecomposition failures
- Reports baseline fidelity, model fidelity, improvement ratio
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training_loop.dataset.StreamingChunkDataset import (
    get_chunk_files,
    split_chunk_files,
    StreamingChunkDataset,
)

# Import models from train_16.models
from train_16.models.transformer_5qubit import HierarchicalTransformer5Qubit
from train_16.models.transformer_8qubit import HierarchicalTransformer8Qubit
from train_16.models.mlp_5qubit import HierarchicalMLP5Qubit
from train_16.models.mlp_5qubit_wide import HierarchicalMLP5QubitWide
from train_16.models.mlp_5qubit_deep import HierarchicalMLP5QubitDeep
from train_16.models.mlp_8qubit import HierarchicalMLP8Qubit


# =============================================================================
# Configuration
# =============================================================================

DTYPE = torch.float64
CDTYPE = torch.complex128

MODEL_REGISTRY_5Q = {
    "transformer_5qubit": HierarchicalTransformer5Qubit,
    "mlp_5qubit": HierarchicalMLP5Qubit,
    "mlp_5qubit_wide": HierarchicalMLP5QubitWide,
    "mlp_5qubit_deep": HierarchicalMLP5QubitDeep,
}

MODEL_REGISTRY_8Q = {
    "transformer_8qubit": HierarchicalTransformer8Qubit,
    "mlp_8qubit": HierarchicalMLP8Qubit,
}


# =============================================================================
# Uhlmann Fidelity (float64)
# =============================================================================


def matrix_sqrt_float64(A: torch.Tensor) -> torch.Tensor:
    """
    Compute matrix square root via eigendecomposition.
    A must be Hermitian positive semidefinite.
    Uses float64/complex128 for numerical stability.

    Returns: sqrt(A), or raises RuntimeError on failure
    """
    # Ensure complex128
    A = A.to(CDTYPE)

    # Eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(A)

    # Check for numerical failures
    if torch.isnan(eigvals).any() or torch.isinf(eigvals).any():
        raise RuntimeError("NaN/Inf in eigenvalues")

    # Clamp small negative eigenvalues (numerical noise)
    eigvals = torch.clamp(eigvals.real, min=0.0)
    sqrt_eigvals = torch.sqrt(eigvals)

    # Reconstruct: V @ diag(sqrt(lambda)) @ V^H
    sqrt_diag = torch.diag(sqrt_eigvals.to(CDTYPE))
    return eigvecs @ sqrt_diag @ eigvecs.mH


def uhlmann_fidelity_single(rho: torch.Tensor, sigma: torch.Tensor) -> float:
    """
    Compute Uhlmann fidelity between two density matrices.

    F(rho, sigma) = (Tr[sqrt(sqrt(rho) @ sigma @ sqrt(rho))])^2

    Args:
        rho: (d, d) complex tensor
        sigma: (d, d) complex tensor

    Returns:
        Fidelity in [0, 1], or raises RuntimeError on failure
    """
    # Ensure Hermitian
    rho = (rho + rho.mH) / 2
    sigma = (sigma + sigma.mH) / 2

    # Normalize trace to 1 (required for valid density matrices)
    # This is necessary because Frobenius loss is scale-invariant,
    # so models may output matrices with arbitrary trace
    rho = rho / torch.trace(rho)
    sigma = sigma / torch.trace(sigma)

    # sqrt(rho)
    sqrt_rho = matrix_sqrt_float64(rho)

    # sqrt(rho) @ sigma @ sqrt(rho)
    inner = sqrt_rho @ sigma @ sqrt_rho

    # Ensure Hermitian
    inner = (inner + inner.mH) / 2

    # sqrt(inner)
    sqrt_inner = matrix_sqrt_float64(inner)

    # F = (Tr[sqrt_inner])^2
    trace_val = torch.trace(sqrt_inner).real
    fid = (trace_val**2).item()

    # Clamp to valid range
    return max(0.0, min(1.0, fid))


def tensor_to_complex(t: torch.Tensor) -> torch.Tensor:
    """Convert (2, H, W) real tensor to (H, W) complex tensor."""
    return t[0].to(CDTYPE) + 1j * t[1].to(CDTYPE)


# =============================================================================
# Self-Check
# =============================================================================


def self_check_fidelity(device: torch.device, matrix_size: int = 32) -> bool:
    """
    Verify F(rho, rho) = 1.0 for random density matrices.
    Returns True if self-check passes.
    """
    print(f"Running self-check: F(rho, rho) should equal 1.0 (d={matrix_size})...")

    # Generate random density matrix via Wishart distribution
    G = torch.randn(matrix_size, matrix_size, dtype=CDTYPE, device=device)
    rho = G @ G.mH
    rho = rho / torch.trace(rho)  # Normalize

    try:
        fid = uhlmann_fidelity_single(rho, rho)
    except RuntimeError as e:
        print(f"  FAILED: Exception during self-check: {e}")
        return False

    # Should be very close to 1.0
    tolerance = 1e-10
    if abs(fid - 1.0) < tolerance:
        print(f"  PASSED: F(rho, rho) = {fid:.15f}")
        return True
    else:
        print(f"  FAILED: F(rho, rho) = {fid:.15f} (expected 1.0, tol={tolerance})")
        return False


# =============================================================================
# Streaming Evaluation Functions
# =============================================================================


@torch.no_grad()
def evaluate_baseline_fidelity_streaming(
    test_files: list,
    device: torch.device,
    batch_size: int = 32,
    samples_per_chunk: int = 1000,
) -> dict:
    """
    Compute baseline Uhlmann fidelity: F(noisy, clean).
    Streams chunks one at a time to avoid OOM.

    Returns:
        dict with 'mean', 'std', 'n_samples', 'n_failures'
    """
    all_fidelities = []
    n_failures = 0
    n_total = 0

    # Use streaming dataset with no shuffle for reproducibility
    dataset = StreamingChunkDataset(
        test_files, shuffle=False, samples_per_chunk=samples_per_chunk, seed=42
    )
    loader = DataLoader(dataset, batch_size=batch_size)

    total_samples = len(test_files) * samples_per_chunk
    pbar = tqdm(
        loader, total=total_samples // batch_size, desc="Baseline F(noisy, clean)"
    )

    for x, y in pbar:
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

        # Update progress bar
        if len(all_fidelities) > 0:
            current_mean = sum(all_fidelities) / len(all_fidelities)
            pbar.set_postfix({"mean": f"{current_mean:.4f}", "failures": n_failures})

    if len(all_fidelities) == 0:
        return {"mean": 0.0, "std": 0.0, "n_samples": 0, "n_failures": n_total}

    fids = torch.tensor(all_fidelities, dtype=DTYPE)
    return {
        "mean": fids.mean().item(),
        "std": fids.std().item(),
        "n_samples": len(all_fidelities),
        "n_failures": n_failures,
    }


@torch.no_grad()
def evaluate_model_fidelity_streaming(
    model: torch.nn.Module,
    test_files: list,
    device: torch.device,
    batch_size: int = 4,
    samples_per_chunk: int = 1000,
) -> dict:
    """
    Compute model Uhlmann fidelity: F(pred, clean).
    Streams chunks one at a time to avoid OOM.

    Returns:
        dict with 'mean', 'std', 'n_samples', 'n_failures'
    """
    model.eval()
    all_fidelities = []
    n_failures = 0
    n_total = 0

    # Use streaming dataset with no shuffle for reproducibility
    dataset = StreamingChunkDataset(
        test_files, shuffle=False, samples_per_chunk=samples_per_chunk, seed=42
    )
    loader = DataLoader(dataset, batch_size=batch_size)

    total_samples = len(test_files) * samples_per_chunk
    pbar = tqdm(loader, total=total_samples // batch_size, desc="Model F(pred, clean)")

    for x, y in pbar:
        x = x.to(device, dtype=DTYPE)
        y = y.to(device, dtype=DTYPE)

        pred = model(x)

        for i in range(pred.shape[0]):
            n_total += 1
            pred_mat = tensor_to_complex(pred[i])
            clean_mat = tensor_to_complex(y[i])

            try:
                fid = uhlmann_fidelity_single(pred_mat, clean_mat)
                all_fidelities.append(fid)
            except RuntimeError:
                n_failures += 1

        # Update progress bar
        if len(all_fidelities) > 0:
            current_mean = sum(all_fidelities) / len(all_fidelities)
            pbar.set_postfix({"mean": f"{current_mean:.4f}", "failures": n_failures})

    if len(all_fidelities) == 0:
        return {"mean": 0.0, "std": 0.0, "n_samples": 0, "n_failures": n_total}

    fids = torch.tensor(all_fidelities, dtype=DTYPE)
    return {
        "mean": fids.mean().item(),
        "std": fids.std().item(),
        "n_samples": len(all_fidelities),
        "n_failures": n_failures,
    }


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Streaming Uhlmann fidelity evaluation for large datasets"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., transformer_8qubit, mlp_8qubit)",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--skip-self-check",
        action="store_true",
        help="Skip F(rho, rho) = 1.0 self-check",
    )
    parser.add_argument(
        "--samples-per-chunk",
        type=int,
        default=1000,
        help="Number of samples per chunk file",
    )
    args = parser.parse_args()

    # Device
    if not torch.cuda.is_available():
        print("FATAL ERROR: No GPU found. Evaluation requires CUDA.")
        sys.exit(1)
    device = torch.device("cuda")
    print(f"Device: {device} ({torch.cuda.get_device_name()})")
    print(f"Dtype: {DTYPE}")

    # Determine model registry and matrix size
    if args.model in MODEL_REGISTRY_5Q:
        ModelClass = MODEL_REGISTRY_5Q[args.model]
        matrix_size = 32  # 5-qubit
    elif args.model in MODEL_REGISTRY_8Q:
        ModelClass = MODEL_REGISTRY_8Q[args.model]
        matrix_size = 256  # 8-qubit
    else:
        all_models = list(MODEL_REGISTRY_5Q.keys()) + list(MODEL_REGISTRY_8Q.keys())
        print(f"Unknown model: {args.model}")
        print(f"Available models: {all_models}")
        sys.exit(1)

    # Self-check
    if not args.skip_self_check:
        if not self_check_fidelity(device, matrix_size=matrix_size):
            print("FATAL: Self-check failed. Aborting evaluation.")
            sys.exit(1)

    # Get chunk files and split
    print(f"\nLoading chunk files from {args.dataset}...")
    chunk_files = get_chunk_files(args.dataset)
    print(f"  Found {len(chunk_files)} chunk files")

    _, _, test_files = split_chunk_files(chunk_files, 0.8, 0.1, seed=42)
    print(
        f"  Test set: {len(test_files)} chunks ({len(test_files) * args.samples_per_chunk} samples)"
    )

    # Compute baseline fidelity
    print("\n" + "=" * 60)
    print("BASELINE: F(noisy, clean)")
    print("=" * 60)
    baseline = evaluate_baseline_fidelity_streaming(
        test_files, device, batch_size=32, samples_per_chunk=args.samples_per_chunk
    )
    print(f"  Mean:     {baseline['mean']:.6f}")
    print(f"  Std:      {baseline['std']:.6f}")
    print(f"  Samples:  {baseline['n_samples']}")
    print(f"  Failures: {baseline['n_failures']}")

    # Load model
    print("\n" + "=" * 60)
    print(f"MODEL: {args.model}")
    print("=" * 60)

    model = ModelClass(dtype=DTYPE).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Handle both direct state_dict and wrapped checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint (epoch {checkpoint.get('epoch', '?')})")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded checkpoint (direct state_dict)")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Evaluate model
    print("\nEvaluating model fidelity: F(pred, clean)...")
    model_result = evaluate_model_fidelity_streaming(
        model,
        test_files,
        device,
        batch_size=args.batch_size,
        samples_per_chunk=args.samples_per_chunk,
    )

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Metric':<20} {'Baseline':<15} {'Model':<15}")
    print("-" * 60)
    print(
        f"{'Mean Fidelity':<20} {baseline['mean']:<15.6f} {model_result['mean']:<15.6f}"
    )
    print(f"{'Std Fidelity':<20} {baseline['std']:<15.6f} {model_result['std']:<15.6f}")
    print(
        f"{'N Samples':<20} {baseline['n_samples']:<15} {model_result['n_samples']:<15}"
    )
    print(
        f"{'N Failures':<20} {baseline['n_failures']:<15} {model_result['n_failures']:<15}"
    )

    # Improvement ratio
    if baseline["mean"] > 0:
        improvement = model_result["mean"] / baseline["mean"]
        print(f"\nImprovement ratio: {improvement:.2f}x")

    # Absolute improvement
    delta = model_result["mean"] - baseline["mean"]
    print(f"Absolute improvement: {delta:+.6f}")

    # Failure rate
    if model_result["n_samples"] + model_result["n_failures"] > 0:
        fail_rate = model_result["n_failures"] / (
            model_result["n_samples"] + model_result["n_failures"]
        )
        print(f"Model failure rate: {fail_rate:.4%}")


if __name__ == "__main__":
    main()
