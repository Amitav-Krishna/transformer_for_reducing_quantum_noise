"""
Evaluate v7 residual MLP on Uhlmann fidelity.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from training_loop.dataset.ChunkDataset import ChunkDataset
from training_loop.dataset.load_chunks import load_chunks
from training_loop.dataset.split_chunks import split_chunks

from train_v7.mlp import MLPResidualAutoencoder
from losses.frob import FrobeniusFidelityLoss


def project_to_density_matrix(rho):
    """Project a matrix to the nearest valid density matrix."""
    rho = (rho + rho.mH) / 2
    rho = rho + 1e-10 * torch.eye(rho.shape[-1], dtype=rho.dtype, device=rho.device)
    try:
        eigvals, eigvecs = torch.linalg.eigh(rho)
    except torch._C._LinAlgError:
        n = rho.shape[-1]
        return torch.eye(n, dtype=rho.dtype, device=rho.device) / n
    eigvals = torch.clamp(eigvals.real, min=0)
    rho = eigvecs @ torch.diag_embed(eigvals.to(eigvecs.dtype)) @ eigvecs.mH
    trace = torch.trace(rho).real
    if trace.abs() > 1e-10:
        rho = rho / trace
    return rho


def matrix_sqrt(A):
    """Compute matrix square root via eigendecomposition."""
    A = A + 1e-10 * torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
    try:
        eigvals, eigvecs = torch.linalg.eigh(A)
    except torch._C._LinAlgError:
        return torch.eye(A.shape[-1], dtype=A.dtype, device=A.device) * 0.1
    eigvals = torch.clamp(eigvals.real, min=1e-12)
    sqrt_eigvals = torch.sqrt(eigvals)
    sqrt_diag = torch.diag_embed(sqrt_eigvals.to(eigvecs.dtype))
    return eigvecs @ sqrt_diag @ eigvecs.mH


def uhlmann_fidelity_batch(rho_pred, rho_true, project=True):
    """Compute Uhlmann fidelity: F(ρ, σ) = (Tr[√(√ρ σ √ρ)])²"""
    rho = rho_pred[:, 0] + 1j * rho_pred[:, 1]
    sigma = rho_true[:, 0] + 1j * rho_true[:, 1]

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
def evaluate_uhlmann_on_chunks(model, chunks, device, batch_size):
    """Compute mean Uhlmann fidelity over all chunks."""
    model.eval()
    all_fid = []

    for X, Y in chunks:
        ds = ChunkDataset(X, Y, "mlp")
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            fid = uhlmann_fidelity_batch(pred, y, project=True)
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

    ckpt_path = "checkpoints_7/mlp/best.pt"

    print(f"\n{'='*50}")
    print("Evaluating Residual MLP (v7)")
    print(f"{'='*50}")

    model = MLPResidualAutoencoder(loss_fn=FrobeniusFidelityLoss()).to(device)
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    except FileNotFoundError:
        print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
        return

    mean_fid, std_fid = evaluate_uhlmann_on_chunks(model, test_chunks, device, 64)

    print(f"  Uhlmann Fidelity: {mean_fid:.4f} ± {std_fid:.4f}")

    print(f"\n{'='*50}")
    print("COMPARISON")
    print(f"{'='*50}")
    print(f"Baseline (noisy vs clean): 0.0682 ± 0.0633")
    print(f"MLP v6 (no residual):      0.0375 ± 0.0089")
    print(f"Transformer v6:            0.1719 ± 0.1591")
    print(f"Residual MLP v7:           {mean_fid:.4f} ± {std_fid:.4f}")


if __name__ == "__main__":
    main()
