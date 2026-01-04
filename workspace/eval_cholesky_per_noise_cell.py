"""
Evaluate Cholesky models per noise type and level.

Since outputs are valid density matrices by construction,
we don't need to symmetrize them before computing fidelity.

Output: csvs_3/noise_cells/{model_name}_noise_cells.csv
"""

import os
import torch
import pandas as pd

from training_loop.dataset.load_chunks import load_chunks_with_metadata
from training_loop.dataset.split_chunks import split_chunks
from training_loop.dataset.ChunkDataset import ChunkDataset
from torch.utils.data import DataLoader

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
def evaluate_model_by_noise(model, test_chunks, arch, device, batch_size):
    """
    Returns:
        stats[noise_type][noise_level] = {mean, std, count}
    """
    stats = {}
    model.eval()

    for blob in test_chunks:
        X = blob["X"]
        Y = blob["Y"]
        meta_list = blob["meta"]   # aligned list of dicts

        ds = ChunkDataset(X, Y, arch)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        batch_index = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            # Cholesky outputs are valid density matrices - no symmetrization needed
            fid = uhlmann_fidelity_batch(pred, yb, symmetrize=False).cpu().numpy()

            # match metadata slice
            start = batch_index * batch_size
            end = start + len(fid)
            meta_slice = meta_list[start:end]

            for f, m in zip(fid, meta_slice):
                ntype = m["noise_type"]
                nlevel = m["noise_level"]

                if ntype not in stats:
                    stats[ntype] = {}
                if nlevel not in stats[ntype]:
                    stats[ntype][nlevel] = {"sum": 0.0, "sum_sq": 0.0, "count": 0}

                cell = stats[ntype][nlevel]
                cell["sum"] += float(f)
                cell["sum_sq"] += float(f) ** 2
                cell["count"] += 1

            batch_index += 1

    # finalize mean/std
    for ntype in stats:
        for nlevel in stats[ntype]:
            cell = stats[ntype][nlevel]
            c = cell["count"]
            mean = cell["sum"] / c
            var = (cell["sum_sq"] / c) - mean * mean
            std = var**0.5 if var > 0 else 0
            stats[ntype][nlevel] = {"mean": mean, "std": std, "count": c}

    model.train()
    return stats


def save_stats_to_csv(stats, out_path):
    rows = []
    for ntype in stats:
        for nlevel in stats[ntype]:
            d = stats[ntype][nlevel]
            rows.append({
                "noise_type": ntype,
                "noise_level": nlevel,
                "mean_fidelity": d["mean"],
                "std_fidelity": d["std"],
                "count": d["count"],
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["noise_type", "noise_level"])
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load dataset once
    print("Loading dataset with metadata...")
    chunks = load_chunks_with_metadata("dataset_smaller")
    _, _, test_chunks = split_chunks(chunks, seed=42)

    out_dir = "csvs_3/noise_cells"
    os.makedirs(out_dir, exist_ok=True)

    # Evaluate every Cholesky model
    for model_name, cfg in EXPERIMENTS.items():
        print(f"\n===== Evaluating model: {model_name} =====")

        arch = cfg["arch"]
        if arch == "mlp":
            batch = 64
        else:
            batch = 8

        # load model
        model = cfg["create_model"]().to(device)
        ckpt_path = f"checkpoints_3/{model_name}/best.pt"

        if not os.path.exists(ckpt_path):
            print(f"[WARNING] No checkpoint found for {model_name}, skipping.")
            continue

        print(f"Loading checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        # evaluate per noise cell
        stats = evaluate_model_by_noise(
            model, test_chunks, arch, device, batch
        )

        # save CSV
        out_path = f"{out_dir}/{model_name}_noise_cells.csv"
        save_stats_to_csv(stats, out_path)

    print("\n==== DONE. All Cholesky models evaluated per noise cell. ====")


if __name__ == "__main__":
    main()