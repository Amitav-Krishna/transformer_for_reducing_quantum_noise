"""
Evaluate train_13 element-wise Cholesky Transformer on Uhlmann fidelity.

Compares against baseline (noisy vs clean).

Usage:
    python train_13/eval_uhlmann.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from scipy.linalg import sqrtm

from train_13.cholesky.transformer_cholesky_elementwise import TransformerCholeskyElementwise
from losses.frob import FrobeniusFidelityLoss

from training_loop.dataset.load_chunks import load_chunks
from training_loop.dataset.split_chunks import split_chunks


def uhlmann_fidelity(rho, sigma, debug=False):
    """Compute Uhlmann fidelity F(rho, sigma) = (Tr[sqrt(sqrt(rho) sigma sqrt(rho))])^2"""
    try:
        sqrt_rho = sqrtm(rho)
        inner = sqrt_rho @ sigma @ sqrt_rho
        sqrt_inner = sqrtm(inner)
        fid = np.real(np.trace(sqrt_inner)) ** 2
        if debug:
            print(f"  rho trace: {np.trace(rho):.4f}, sigma trace: {np.trace(sigma):.4f}")
            print(f"  fid: {fid:.4f}")
        return np.clip(fid, 0, 1)
    except Exception as e:
        if debug:
            print(f"  Error: {e}")
        return 0.0


def evaluate_model(model, test_chunks, device, model_name):
    """Evaluate a model on Uhlmann fidelity."""
    model.eval()

    fidelities = []
    baseline_fidelities = []
    first_sample = True

    with torch.no_grad():
        for chunk_idx, (X, Y) in enumerate(test_chunks):
            batch_size = X.shape[0]

            # Raw chunks are (N, 32, 32, 2), need to permute to (N, 2, 32, 32)
            X_perm = X.permute(0, 3, 1, 2).float().to(device)
            Y_perm = Y.permute(0, 3, 1, 2).float()

            preds = model(X_perm).cpu().numpy()

            for i in range(batch_size):
                # Get density matrices - data is now (2, 32, 32) after permute
                noisy_dm = X_perm[i, 0].cpu().numpy() + 1j * X_perm[i, 1].cpu().numpy()
                clean_dm = Y_perm[i, 0].numpy() + 1j * Y_perm[i, 1].numpy()
                pred_dm = preds[i, 0] + 1j * preds[i, 1]

                # Debug first sample
                if first_sample:
                    print(f"  First sample debug:")
                    print(f"    X shape: {X.shape}, X_perm shape: {X_perm.shape}")
                    print(f"    noisy_dm trace: {np.trace(noisy_dm):.4f}")
                    print(f"    clean_dm trace: {np.trace(clean_dm):.4f}")
                    print(f"    pred_dm trace: {np.trace(pred_dm):.4f}")
                    print(f"    pred range: [{preds[i].min():.4f}, {preds[i].max():.4f}]")

                # Compute fidelities
                model_fid = uhlmann_fidelity(pred_dm, clean_dm, debug=first_sample)
                baseline_fid = uhlmann_fidelity(noisy_dm, clean_dm, debug=first_sample)

                if first_sample:
                    print(f"    model_fid: {model_fid:.4f}, baseline_fid: {baseline_fid:.4f}")
                    first_sample = False

                fidelities.append(model_fid)
                baseline_fidelities.append(baseline_fid)

            if (chunk_idx + 1) % 10 == 0:
                print(f"  Processed {chunk_idx + 1}/{len(test_chunks)} chunks...")

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

    # Load test data
    print("Loading dataset...")
    chunks = load_chunks("dataset_smaller")
    _, _, test_chunks = split_chunks(chunks)
    print(f"Test chunks: {len(test_chunks)}")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Evaluate Element-wise Cholesky Transformer
    print("\n" + "="*60)
    print("Evaluating Element-wise Cholesky Transformer...")
    model = TransformerCholeskyElementwise(loss_fn=FrobeniusFidelityLoss()).to(device)
    model.load_state_dict(torch.load(
        f"{base_dir}/checkpoints_13/transformer_cholesky_elementwise/best.pt",
        map_location=device
    ))
    fid, std, baseline = evaluate_model(model, test_chunks, device, "Transformer Cholesky (Element-wise)")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Model':<35} {'Fidelity':>12} {'Improvement':>12}")
    print("-"*60)
    print(f"{'Baseline (noisy)':<35} {baseline:>12.4f} {1.0:>12.2f}x")
    print(f"{'Transformer Cholesky (Element-wise)':<35} {fid:>12.4f} {fid/baseline:>12.2f}x")

    # Comparison with previous results
    print("\n" + "="*60)
    print("COMPARISON WITH PREVIOUS RESULTS")
    print("="*60)
    print("Post-hoc Transformer:            0.28 (2.3x)")
    print("Global-pooling Cholesky Trans:   0.032 (0.26x)")
    print(f"Element-wise Cholesky Trans:     {fid:.3f} ({fid/baseline:.2f}x)")


if __name__ == "__main__":
    main()
