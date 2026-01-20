"""
Evaluate transformer_matched_v2 models on Uhlmann fidelity.

Run after training completes:
    python eval_transformer_matched_v2_uhlmann.py

Outputs:
    - Prints mean Uhlmann fidelity for both variants
    - Saves to csvs_2/transformer_matched_v2_uhlmann_results.csv
"""

import torch
import pandas as pd
import os

from models.transformer_matched_v2 import TransformerAutoencoderMatchedV2
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
    # Convert to complex and transpose to (B, H, W)
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


def evaluate_model(model, test_chunks, device):
    """Evaluate model on test set, return mean Uhlmann fidelity."""
    model.eval()
    model.to(device)

    all_fidelities = []

    for blob in test_chunks:
        X = blob["X"].to(device)
        Y = blob["Y"].to(device)

        # Model expects (B, 2, 32, 32), Y is (B, 32, 32, 2)
        if X.shape[-1] == 2:
            X = X.permute(0, 3, 1, 2)
        if Y.shape[-1] == 2:
            Y = Y.permute(0, 3, 1, 2)

        with torch.no_grad():
            pred = model(X)

        fids = uhlmann_fidelity(pred.cpu().double(), Y.cpu().double())
        all_fidelities.extend(fids.tolist())

    return all_fidelities


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test data
    chunks = load_chunks_with_metadata("dataset_smaller")
    _, _, test_chunks = split_chunks(chunks, seed=42)

    results = []

    models_to_eval = {
        "transformer_matched_v2_frob": FrobeniusFidelityLoss(),
        "transformer_matched_v2_physics": CompositePhysicsTotalLoss(),
    }

    for name, loss_fn in models_to_eval.items():
        checkpoint_path = f"checkpoints_2/{name}/best.pt"

        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}, skipping...")
            continue

        print(f"\nEvaluating {name}...")

        model = TransformerAutoencoderMatchedV2(loss_fn=loss_fn)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)

        fidelities = evaluate_model(model, test_chunks, device)

        mean_fid = sum(fidelities) / len(fidelities)
        std_fid = (sum((f - mean_fid) ** 2 for f in fidelities) / len(fidelities)) ** 0.5

        print(f"  Uhlmann Fidelity: {mean_fid:.4f} ± {std_fid:.4f}")

        results.append({
            "model": name,
            "mean_fidelity": mean_fid,
            "std_fidelity": std_fid,
            "n_samples": len(fidelities),
        })

    # Save results
    if results:
        df = pd.DataFrame(results)
        os.makedirs("csvs_2", exist_ok=True)
        df.to_csv("csvs_2/transformer_matched_v2_uhlmann_results.csv", index=False)
        print(f"\nResults saved to csvs_2/transformer_matched_v2_uhlmann_results.csv")


if __name__ == "__main__":
    main()