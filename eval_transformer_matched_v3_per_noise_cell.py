"""
Evaluate transformer_matched_v3 models per noise type and level.

Run after training completes:
    python eval_transformer_matched_v3_per_noise_cell.py

Outputs:
    - csvs_2/noise_cells/transformer_matched_v3_frob_noise_cells.csv
    - csvs_2/noise_cells/transformer_matched_v3_physics_noise_cells.csv
"""

import torch
import pandas as pd
import os
from collections import defaultdict

from models.transformer_matched_v3 import TransformerAutoencoderMatchedV3
from losses.frob import FrobeniusFidelityLoss
from losses.total_physics_loss import CompositePhysicsTotalLoss
from training_loop.dataset.load_chunks import load_chunks_with_metadata
from training_loop.dataset.split_chunks import split_chunks


@torch.no_grad()
def matrix_sqrt(m):
    evals, evecs = torch.linalg.eigh(m)
    evals = torch.clamp(evals, min=0)
    return (evecs * torch.sqrt(evals)[..., None, :]) @ evecs.conj().transpose(-1, -2)


@torch.no_grad()
def uhlmann_fidelity(rho, sigma):
    """
    Compute Uhlmann fidelity F(rho, sigma) = (Tr[sqrt(sqrt(rho) sigma sqrt(rho))])^2
    rho, sigma: (B, 2, H, W) with channels [real, imag]
    returns: (B,) fidelities
    """
    rho_c = rho[:, 0] + 1j * rho[:, 1]
    sig_c = sigma[:, 0] + 1j * sigma[:, 1]

    # Symmetrize for Hermiticity
    rho_c = (rho_c + rho_c.conj().transpose(-1, -2)) / 2
    sig_c = (sig_c + sig_c.conj().transpose(-1, -2)) / 2

    sr = matrix_sqrt(rho_c)
    inner = sr @ sig_c @ sr
    s_inner = matrix_sqrt(inner)

    tr = torch.real(torch.diagonal(s_inner, dim1=-2, dim2=-1).sum(-1))
    return tr ** 2


def evaluate_model_per_cell(model, test_chunks, device):
    """Evaluate model on test set, return fidelities grouped by noise type/level."""
    model.eval()
    model.to(device)

    # Group fidelities by (noise_type, noise_level)
    cell_fidelities = defaultdict(list)

    for blob in test_chunks:
        X = blob["X"].to(device)
        Y = blob["Y"].to(device)
        meta = blob["meta"]

        # Model expects (B, 2, 32, 32), Y is (B, 32, 32, 2)
        if X.shape[-1] == 2:
            X = X.permute(0, 3, 1, 2)
        if Y.shape[-1] == 2:
            Y = Y.permute(0, 3, 1, 2)

        with torch.no_grad():
            pred = model(X)

        fids = uhlmann_fidelity(pred.cpu().double(), Y.cpu().double())

        for fid, m in zip(fids.tolist(), meta):
            key = (m["noise_type"], m["noise_level"])
            cell_fidelities[key].append(fid)

    return cell_fidelities


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test data with metadata
    chunks = load_chunks_with_metadata("dataset_smaller")
    _, _, test_chunks = split_chunks(chunks, seed=42)

    os.makedirs("csvs_2/noise_cells", exist_ok=True)

    models_to_eval = {
        "transformer_matched_v3_frob": FrobeniusFidelityLoss(),
        "transformer_matched_v3_physics": CompositePhysicsTotalLoss(),
    }

    for name, loss_fn in models_to_eval.items():
        checkpoint_path = f"checkpoints_2/{name}/best.pt"

        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}, skipping...")
            continue

        print(f"\nEvaluating {name}...")

        model = TransformerAutoencoderMatchedV3(loss_fn=loss_fn)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)

        cell_fidelities = evaluate_model_per_cell(model, test_chunks, device)

        # Aggregate to DataFrame
        rows = []
        for (noise_type, noise_level), fids in cell_fidelities.items():
            mean_fid = sum(fids) / len(fids)
            std_fid = (sum((f - mean_fid) ** 2 for f in fids) / len(fids)) ** 0.5
            rows.append({
                "noise_type": noise_type,
                "noise_level": noise_level,
                "mean_fidelity": mean_fid,
                "std_fidelity": std_fid,
                "count": len(fids),
            })

        df = pd.DataFrame(rows)
        df = df.sort_values(["noise_type", "noise_level"])

        out_path = f"csvs_2/noise_cells/{name}_noise_cells.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")

        # Print summary
        overall_mean = df["mean_fidelity"].mean()
        print(f"  Overall mean fidelity: {overall_mean:.4f}")


if __name__ == "__main__":
    main()