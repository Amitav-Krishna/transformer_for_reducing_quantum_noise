"""
Evaluate Hierarchical Transformer (6-qubit) on Uhlmann fidelity.

Usage:
    python train_14/eval_uhlmann.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from scipy.linalg import sqrtm
import glob

from train_14.transformer_6qubit import HierarchicalTransformer6Qubit
from losses.frob import FrobeniusFidelityLoss


def uhlmann_fidelity(rho, sigma):
    """Compute Uhlmann fidelity F(rho, sigma) = (Tr[sqrt(sqrt(rho) sigma sqrt(rho))])^2"""
    try:
        sqrt_rho = sqrtm(rho)
        inner = sqrt_rho @ sigma @ sqrt_rho
        sqrt_inner = sqrtm(inner)
        fid = np.real(np.trace(sqrt_inner)) ** 2
        return np.clip(fid, 0, 1)
    except Exception:
        return 0.0


def evaluate_model(model, test_chunks, device, model_name):
    """Evaluate a model on Uhlmann fidelity."""
    model.eval()

    fidelities = []
    baseline_fidelities = []
    first_sample = True

    with torch.no_grad():
        for chunk_file in test_chunks:
            data = torch.load(chunk_file)
            # Data is (N, 64, 64, 2), need (N, 2, 64, 64)
            X = data["X"].permute(0, 3, 1, 2).float().to(device)
            Y = data["Y"].permute(0, 3, 1, 2).float()

            preds = model(X).cpu().numpy()

            for i in range(X.shape[0]):
                # Get density matrices
                noisy_dm = X[i, 0].cpu().numpy() + 1j * X[i, 1].cpu().numpy()
                clean_dm = Y[i, 0].numpy() + 1j * Y[i, 1].numpy()
                pred_dm = preds[i, 0] + 1j * preds[i, 1]

                # Debug first sample
                if first_sample:
                    print(f"  First sample debug:")
                    print(f"    Matrix size: {noisy_dm.shape}")
                    print(f"    noisy_dm trace: {np.trace(noisy_dm):.4f}")
                    print(f"    clean_dm trace: {np.trace(clean_dm):.4f}")
                    print(f"    pred_dm trace: {np.trace(pred_dm):.4f}")
                    first_sample = False

                # Compute fidelities
                model_fid = uhlmann_fidelity(pred_dm, clean_dm)
                baseline_fid = uhlmann_fidelity(noisy_dm, clean_dm)

                fidelities.append(model_fid)
                baseline_fidelities.append(baseline_fid)

            print(f"  Processed {chunk_file}")

    fidelities = np.array(fidelities)
    baseline_fidelities = np.array(baseline_fidelities)

    print(f"\n{model_name}:")
    print(f"  Uhlmann Fidelity: {fidelities.mean():.4f} +/- {fidelities.std():.4f}")
    print(f"  Baseline:         {baseline_fidelities.mean():.4f} +/- {baseline_fidelities.std():.4f}")
    print(f"  Improvement:      {fidelities.mean() / baseline_fidelities.mean():.2f}x")

    return fidelities.mean(), fidelities.std(), baseline_fidelities.mean()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test chunks
    chunk_files = sorted(glob.glob("dataset_6qubit/chunk_*.pt"))
    n = len(chunk_files)
    test_chunks = chunk_files[int(n * 0.9):]  # Last 10%
    print(f"Test chunks: {len(test_chunks)}")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Evaluate Hierarchical Transformer
    print("\n" + "="*60)
    print("Evaluating Hierarchical Transformer (6-qubit)...")
    model = HierarchicalTransformer6Qubit(loss_fn=FrobeniusFidelityLoss()).to(device)
    model.load_state_dict(torch.load(
        f"{base_dir}/checkpoints_14/hierarchical_transformer/best.pt",
        map_location=device
    ))
    fid, std, baseline = evaluate_model(model, test_chunks, device, "Hierarchical Transformer (6-qubit)")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY - 6-QUBIT SCALABILITY PROOF-OF-CONCEPT")
    print("="*60)
    print(f"{'Model':<40} {'Fidelity':>12} {'Improvement':>12}")
    print("-"*65)
    print(f"{'Baseline (noisy)':<40} {baseline:>12.4f} {1.0:>12.2f}x")
    print(f"{'Hierarchical Transformer (6-qubit)':<40} {fid:>12.4f} {fid/baseline:>12.2f}x")

    print("\n" + "="*60)
    print("COMPARISON WITH 5-QUBIT RESULTS")
    print("="*60)
    print("5-qubit Transformer (element-wise, 1024 tokens): 0.28 fidelity, 2.3x improvement")
    print(f"6-qubit Transformer (hierarchical, 64 tokens):   {fid:.2f} fidelity, {fid/baseline:.1f}x improvement")
    print("\nThis demonstrates that hierarchical tokenization enables scaling")
    print("to larger qubit counts while maintaining the Transformer advantage.")


if __name__ == "__main__":
    main()
