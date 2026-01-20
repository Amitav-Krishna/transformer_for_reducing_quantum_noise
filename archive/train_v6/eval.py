"""
Evaluate v6 models on Uhlmann fidelity with post-hoc projection.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from training_loop.dataset.ChunkDataset import ChunkDataset
from training_loop.dataset.load_chunks import load_chunks
from training_loop.dataset.split_chunks import split_chunks

from train_v6.mlp import MLPAutoencoder
from train_v6.transformer import TransformerAutoencoder
from losses.frob import FrobeniusFidelityLoss


EXPERIMENTS = {
    "mlp": {
        "arch": "mlp",
        "create_model": lambda: MLPAutoencoder(loss_fn=FrobeniusFidelityLoss())
    },
    "transformer": {
        "arch": "transformer",
        "create_model": lambda: TransformerAutoencoder(loss_fn=FrobeniusFidelityLoss())
    },
}


def project_to_density_matrix(rho):
    """
    Project a matrix to the nearest valid density matrix.
    """
    # 1. Hermitianize
    rho = (rho + rho.mH) / 2

    # 2. Project to PSD via eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(rho)
    eigvals = torch.clamp(eigvals, min=0)
    rho = eigvecs @ torch.diag_embed(eigvals.to(eigvecs.dtype)) @ eigvecs.mH

    # 3. Normalize trace
    trace = torch.trace(rho)
    if trace.abs() > 1e-10:
        rho = rho / trace

    return rho


def matrix_sqrt(A):
    """Compute matrix square root via eigendecomposition."""
    eigvals, eigvecs = torch.linalg.eigh(A)
    eigvals = torch.clamp(eigvals, min=0)
    sqrt_eigvals = torch.sqrt(eigvals)
    sqrt_diag = torch.diag_embed(sqrt_eigvals.to(eigvecs.dtype))
    return eigvecs @ sqrt_diag @ eigvecs.mH


def uhlmann_fidelity_batch(rho_pred, rho_true, project=True):
    """
    Compute Uhlmann fidelity: F(ρ, σ) = (Tr[√(√ρ σ √ρ)])²
    """
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
def evaluate_uhlmann_on_chunks(model, chunks, arch, device, batch_size):
    """Compute mean Uhlmann fidelity over all chunks."""
    model.eval()
    all_fid = []

    for X, Y in chunks:
        ds = ChunkDataset(X, Y, arch)
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

    results = {}

    for name, config in EXPERIMENTS.items():
        arch = config["arch"]
        ckpt_path = f"checkpoints_6/{name}/best.pt"

        print(f"\n{'='*50}")
        print(f"Evaluating {name}")
        print(f"{'='*50}")

        model = config["create_model"]().to(device)
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        except FileNotFoundError:
            print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
            continue

        if arch == "mlp":
            batch_size = 64
        else:
            batch_size = 8

        mean_fid, std_fid = evaluate_uhlmann_on_chunks(
            model, test_chunks, arch, device, batch_size
        )

        results[name] = {"mean": mean_fid, "std": std_fid}
        print(f"  Uhlmann Fidelity: {mean_fid:.4f} ± {std_fid:.4f}")

    # Summary table
    print(f"\n{'='*50}")
    print("SUMMARY: Uhlmann Fidelity on Test Set (v6 Models)")
    print(f"{'='*50}")
    print(f"{'Model':<25} {'Mean':>10} {'Std':>10}")
    print("-" * 50)
    for name, res in results.items():
        print(f"{name:<25} {res['mean']:>10.4f} {res['std']:>10.4f}")


if __name__ == "__main__":
    main()
