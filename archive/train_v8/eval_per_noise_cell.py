"""
Evaluate models per noise cell (noise_type × noise_level).
Outputs CSVs for heatmap generation.

Run on runpod:
    python train_v8/eval_per_noise_cell.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
from collections import defaultdict
from torch.utils.data import DataLoader

from training_loop.dataset.ChunkDataset import ChunkDataset
from training_loop.dataset.load_chunks import load_chunks_with_metadata
from training_loop.dataset.split_chunks import split_chunks

from train_v8.mlp import MLPResidualAutoencoder
from train_v8.transformer import TransformerAutoencoder
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


def uhlmann_fidelity_single(rho, sigma, project=True):
    """Compute Uhlmann fidelity: F(ρ, σ) = (Tr[√(√ρ σ √ρ)])²"""
    if project:
        rho = project_to_density_matrix(rho)
        sigma = project_to_density_matrix(sigma)

    sqrt_rho = matrix_sqrt(rho)
    inner = sqrt_rho @ sigma @ sqrt_rho
    inner = (inner + inner.mH) / 2
    sqrt_inner = matrix_sqrt(inner)

    trace_val = torch.trace(sqrt_inner).real
    fid = trace_val ** 2
    fid = torch.clamp(fid, 0.0, 1.0)
    return fid.item()


def to_complex(x):
    """Convert (B, 2, 32, 32) real tensor to (B, 32, 32) complex tensor."""
    return x[:, 0] + 1j * x[:, 1]


@torch.no_grad()
def evaluate_per_noise_cell(model, chunks_with_meta, device, batch_size, model_type="mlp"):
    """
    Evaluate model and return per-sample fidelities with noise metadata.
    Returns: list of dicts with keys [noise_type, noise_level, fidelity]
    """
    if model is not None:
        model.eval()
    results = []

    for blob in chunks_with_meta:
        X = blob["X"]
        Y = blob["Y"]
        meta = blob["meta"]  # list of dicts with noise_type, noise_level, depth

        ds = ChunkDataset(X, Y, model_type)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        idx = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            B = x.shape[0]

            if model is not None:
                pred = model(x)
            else:
                # Baseline: use noisy input as "prediction"
                pred = x

            pred_complex = to_complex(pred)
            y_complex = to_complex(y)

            for i in range(B):
                fid = uhlmann_fidelity_single(pred_complex[i], y_complex[i], project=True)
                results.append({
                    "noise_type": meta[idx + i]["noise_type"],
                    "noise_level": float(meta[idx + i]["noise_level"]),
                    "fidelity": fid,
                })
            idx += B

    return results


def aggregate_results(results):
    """Aggregate per-sample results into per-cell statistics."""
    cells = defaultdict(list)
    for r in results:
        key = (r["noise_type"], r["noise_level"])
        cells[key].append(r["fidelity"])

    rows = []
    for (ntype, nlevel), fids in cells.items():
        rows.append({
            "noise_type": ntype,
            "noise_level": nlevel,
            "mean_fidelity": sum(fids) / len(fids),
            "std_fidelity": (sum((f - sum(fids)/len(fids))**2 for f in fids) / len(fids)) ** 0.5,
            "count": len(fids),
        })
    return pd.DataFrame(rows)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load chunks with metadata
    chunks = load_chunks_with_metadata("dataset_smaller")

    # Split using indices to preserve metadata
    from training_loop.dataset.split_chunks import split_chunks
    _, _, test_indices = split_chunks(chunks, 0.8, 0.1, 42, return_indices=True)
    test_chunks = [chunks[i] for i in test_indices]

    print(f"Loaded {len(test_chunks)} test chunks")

    # Output directory
    out_dir = "csvs_8/noise_cells"
    os.makedirs(out_dir, exist_ok=True)

    # === Baseline (noisy vs clean) ===
    print("\nEvaluating Baseline (noisy vs clean)...")
    baseline_results = evaluate_per_noise_cell(None, test_chunks, device, 64, "mlp")
    baseline_df = aggregate_results(baseline_results)
    baseline_df.to_csv(f"{out_dir}/baseline_noise_cells.csv", index=False)
    print(f"  Saved: {out_dir}/baseline_noise_cells.csv")
    print(f"  Overall: {sum(r['fidelity'] for r in baseline_results)/len(baseline_results):.4f}")

    # === Residual MLP (v8) ===
    print("\nEvaluating Residual MLP (v8)...")
    mlp = MLPResidualAutoencoder(loss_fn=FrobeniusFidelityLoss()).to(device)
    mlp.load_state_dict(torch.load("checkpoints_8/mlp/best.pt", map_location=device))
    mlp_results = evaluate_per_noise_cell(mlp, test_chunks, device, 64, "mlp")
    mlp_df = aggregate_results(mlp_results)
    mlp_df.to_csv(f"{out_dir}/mlp_noise_cells.csv", index=False)
    print(f"  Saved: {out_dir}/mlp_noise_cells.csv")
    print(f"  Overall: {sum(r['fidelity'] for r in mlp_results)/len(mlp_results):.4f}")

    # === Transformer (v8) ===
    print("\nEvaluating Transformer (v8)...")
    transformer = TransformerAutoencoder(loss_fn=FrobeniusFidelityLoss()).to(device)
    transformer.load_state_dict(torch.load("checkpoints_8/transformer/best.pt", map_location=device))
    transformer_results = evaluate_per_noise_cell(transformer, test_chunks, device, 8, "transformer")
    transformer_df = aggregate_results(transformer_results)
    transformer_df.to_csv(f"{out_dir}/transformer_noise_cells.csv", index=False)
    print(f"  Saved: {out_dir}/transformer_noise_cells.csv")
    print(f"  Overall: {sum(r['fidelity'] for r in transformer_results)/len(transformer_results):.4f}")

    print("\nDone! CSVs saved to:", out_dir)
    print("\nPer-cell summary:")
    print("\nBaseline:")
    print(baseline_df.to_string(index=False))
    print("\nMLP (residual):")
    print(mlp_df.to_string(index=False))
    print("\nTransformer:")
    print(transformer_df.to_string(index=False))


if __name__ == "__main__":
    main()
