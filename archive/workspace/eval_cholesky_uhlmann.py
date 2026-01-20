"""
Evaluate Cholesky models on Uhlmann fidelity.

Since outputs are valid density matrices by construction,
we don't need to symmetrize them before computing fidelity.
"""

import torch
from torch.utils.data import DataLoader

from training_loop.dataset.ChunkDataset import ChunkDataset
from training_loop.dataset.load_chunks import load_chunks
from training_loop.dataset.split_chunks import split_chunks

from models_3.mlp_cholesky import MLPCholeskyAutoencoder
from models_3.transformer_cholesky import TransformerCholeskyAutoencoder
from losses.frob import FrobeniusFidelityLoss


EXPERIMENTS = {
    "mlp_cholesky": {
        "arch": "mlp",
        "create_model": lambda: MLPCholeskyAutoencoder(loss_fn=FrobeniusFidelityLoss())
    },
    "transformer_cholesky": {
        "arch": "transformer",
        "create_model": lambda: TransformerCholeskyAutoencoder(loss_fn=FrobeniusFidelityLoss())
    },
}


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


def uhlmann_fidelity_batch(rho_pred, rho_true, symmetrize=False):
    """
    Compute Uhlmann fidelity between predicted and true density matrices.

    F(ρ, σ) = (Tr[√(√ρ σ √ρ)])²

    Args:
        rho_pred, rho_true: (B, 2, 32, 32) with [real, imag] channels
        symmetrize: If True, symmetrize inputs (not needed for Cholesky outputs)

    Returns: (B,) tensor of fidelities in [0, 1]
    """
    rho = rho_pred[:, 0] + 1j * rho_pred[:, 1]
    sigma = rho_true[:, 0] + 1j * rho_true[:, 1]

    B = rho.shape[0]
    fidelities = []

    for i in range(B):
        r = rho[i]
        s = sigma[i]

        if symmetrize:
            r = (r + r.mH) / 2
            s = (s + s.mH) / 2

        # Compute √ρ
        sqrt_r = matrix_sqrt(r)

        # Compute √ρ σ √ρ
        inner = sqrt_r @ s @ sqrt_r

        # Ensure Hermitian for numerical stability
        inner = (inner + inner.mH) / 2

        # Compute √(√ρ σ √ρ)
        sqrt_inner = matrix_sqrt(inner)

        # F = (Tr[√(√ρ σ √ρ)])²
        trace_val = torch.trace(sqrt_inner).real
        fid = trace_val ** 2

        # Clamp to valid range
        fid = torch.clamp(fid, 0.0, 1.0)
        fidelities.append(fid)

    return torch.stack(fidelities)


@torch.no_grad()
def evaluate_uhlmann_on_chunks(model, chunks, arch, device, batch_size):
    """
    Compute mean Uhlmann fidelity over all chunks.
    Returns: (mean_fidelity, std_fidelity)
    """
    model.eval()
    all_fid = []

    for X, Y in chunks:
        ds = ChunkDataset(X, Y, arch)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            # Cholesky outputs are valid density matrices - no symmetrization needed
            fid = uhlmann_fidelity_batch(pred, y, symmetrize=False)
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
        ckpt_path = f"checkpoints_3/{name}/best.pt"

        print(f"\n{'='*50}")
        print(f"Evaluating {name}")
        print(f"{'='*50}")

        # Load model
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

        # Evaluate Uhlmann fidelity
        mean_fid, std_fid = evaluate_uhlmann_on_chunks(
            model, test_chunks, arch, device, batch_size
        )

        results[name] = {"mean": mean_fid, "std": std_fid}
        print(f"  Uhlmann Fidelity: {mean_fid:.4f} ± {std_fid:.4f}")

    # Summary table
    print(f"\n{'='*50}")
    print("SUMMARY: Uhlmann Fidelity on Test Set (Cholesky Models)")
    print(f"{'='*50}")
    print(f"{'Model':<25} {'Mean':>10} {'Std':>10}")
    print("-" * 50)
    for name, res in results.items():
        print(f"{name:<25} {res['mean']:>10.4f} {res['std']:>10.4f}")


if __name__ == "__main__":
    main()