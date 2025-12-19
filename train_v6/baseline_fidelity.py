"""
Compute baseline Uhlmann fidelity: noisy input vs clean target.

This gives us the "do nothing" baseline to compare against.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from training_loop.dataset.ChunkDataset import ChunkDataset
from training_loop.dataset.load_chunks import load_chunks
from training_loop.dataset.split_chunks import split_chunks


def project_to_density_matrix(rho):
    """Project a matrix to the nearest valid density matrix."""
    rho = (rho + rho.mH) / 2
    # Add regularization for numerical stability
    rho = rho + 1e-10 * torch.eye(rho.shape[-1], dtype=rho.dtype, device=rho.device)
    try:
        eigvals, eigvecs = torch.linalg.eigh(rho)
    except torch._C._LinAlgError:
        # Fallback to maximally mixed state
        n = rho.shape[-1]
        return torch.eye(n, dtype=rho.dtype, device=rho.device) / n
    eigvals = torch.clamp(eigvals.real, min=0)
    rho = eigvecs @ torch.diag_embed(eigvals.to(eigvecs.dtype)) @ eigvecs.mH
    trace = torch.trace(rho).real
    if trace.abs() > 1e-10:
        rho = rho / trace
    return rho


def matrix_sqrt(A):
    """Compute matrix square root via eigendecomposition with numerical stability."""
    # Add small regularization for numerical stability
    A = A + 1e-10 * torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
    try:
        eigvals, eigvecs = torch.linalg.eigh(A)
    except torch._C._LinAlgError:
        # Fallback: return identity-scaled approximation
        return torch.eye(A.shape[-1], dtype=A.dtype, device=A.device) * 0.1
    eigvals = torch.clamp(eigvals.real, min=1e-12)
    sqrt_eigvals = torch.sqrt(eigvals)
    sqrt_diag = torch.diag_embed(sqrt_eigvals.to(eigvecs.dtype))
    return eigvecs @ sqrt_diag @ eigvecs.mH


def uhlmann_fidelity_batch(rho_noisy, rho_clean, project=True):
    """
    Compute Uhlmann fidelity: F(ρ, σ) = (Tr[√(√ρ σ √ρ)])²
    """
    rho = rho_noisy[:, 0] + 1j * rho_noisy[:, 1]
    sigma = rho_clean[:, 0] + 1j * rho_clean[:, 1]

    B = rho.shape[0]
    fidelities = []

    for i in range(B):
        r = rho[i]
        s = sigma[i]

        if project:
            r = project_to_density_matrix(r)
            s = project_to_density_matrix(s)

        sqrt_r = matrix_sqrt(r)
        inner = sqrt_r @ s @ sqrt_r
        inner = (inner + inner.mH) / 2
        sqrt_inner = matrix_sqrt(inner)

        trace_val = torch.trace(sqrt_inner).real
        fid = trace_val ** 2
        fid = torch.clamp(fid, 0.0, 1.0)
        fidelities.append(fid)

    return torch.stack(fidelities)


@torch.no_grad()
def compute_baseline_fidelity(chunks, device, batch_size=64):
    """Compute baseline fidelity: noisy input vs clean target."""
    all_fid = []

    for X, Y in chunks:
        # X = noisy, Y = clean
        ds = ChunkDataset(X, Y, "mlp")  # arch doesn't matter for baseline
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        for noisy, clean in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            # Compare noisy input directly to clean target
            fid = uhlmann_fidelity_batch(noisy, clean, project=True)
            all_fid.append(fid.cpu())

    all_fid = torch.cat(all_fid)
    return all_fid.mean().item(), all_fid.std().item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and split data
    chunks = load_chunks("dataset_smaller")
    _, _, test_chunks = split_chunks(chunks, 0.8, 0.1, 42)
    print(f"Loaded {len(test_chunks)} test chunks")

    print(f"\n{'='*50}")
    print("Computing BASELINE: Noisy Input vs Clean Target")
    print(f"{'='*50}")

    mean_fid, std_fid = compute_baseline_fidelity(test_chunks, device)

    print(f"\nBaseline Uhlmann Fidelity: {mean_fid:.4f} ± {std_fid:.4f}")
    print("\nThis is the 'do nothing' baseline - how similar noisy inputs")
    print("are to clean targets without any model intervention.")


if __name__ == "__main__":
    main()
