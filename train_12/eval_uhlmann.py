"""
Evaluate train_12 models on Uhlmann fidelity.

Compares:
- MLP Cholesky
- Transformer Cholesky
- CNN
- Baseline (noisy vs clean)

Usage:
    python train_12/eval_uhlmann.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from scipy.linalg import sqrtm

from train_12.cholesky.mlp_cholesky import MLPCholeskyAutoencoder
from train_12.cholesky.transformer_cholesky import TransformerCholeskyAutoencoder
from train_12.cnn.cnn import CNNAutoencoder
from losses.frob import FrobeniusFidelityLoss

from training_loop.dataset.load_chunks import load_chunks
from training_loop.dataset.split_chunks import split_chunks


def project_to_valid_dm(rho):
    """Project to valid density matrix (Hermitian, PSD, trace=1)."""
    # Make Hermitian
    rho = (rho + rho.conj().T) / 2
    # Project to PSD
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.maximum(eigvals, 0)
    rho = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
    # Normalize trace
    trace = np.trace(rho)
    if np.abs(trace) > 1e-10:
        rho = rho / trace
    return rho


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


def evaluate_model(model, test_chunks, device, model_name, project_output=False):
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

                # Project to valid density matrix if requested (for CNN/unconstrained models)
                if project_output:
                    pred_dm = project_to_valid_dm(pred_dm)

                # Debug first sample
                if first_sample:
                    print(f"  First sample debug (project_output={project_output}):")
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
    print(f"  Uhlmann Fidelity: {fidelities.mean():.4f} ± {fidelities.std():.4f}")
    print(f"  Baseline:         {baseline_fidelities.mean():.4f} ± {baseline_fidelities.std():.4f}")
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
    results = {}

    # Evaluate MLP Cholesky
    print("\n" + "="*60)
    print("Evaluating MLP Cholesky...")
    mlp_chol = MLPCholeskyAutoencoder(loss_fn=FrobeniusFidelityLoss()).to(device)
    mlp_chol.load_state_dict(torch.load(
        f"{base_dir}/checkpoints_12/cholesky/mlp_cholesky/best.pt",
        map_location=device
    ))
    results['mlp_cholesky'] = evaluate_model(mlp_chol, test_chunks, device, "MLP Cholesky")

    # Evaluate Transformer Cholesky
    print("\n" + "="*60)
    print("Evaluating Transformer Cholesky...")
    trans_chol = TransformerCholeskyAutoencoder(loss_fn=FrobeniusFidelityLoss()).to(device)
    trans_chol.load_state_dict(torch.load(
        f"{base_dir}/checkpoints_12/cholesky/transformer_cholesky/best.pt",
        map_location=device
    ))
    results['transformer_cholesky'] = evaluate_model(trans_chol, test_chunks, device, "Transformer Cholesky")

    # Evaluate CNN (with post-hoc projection for fair comparison)
    print("\n" + "="*60)
    print("Evaluating CNN (with post-hoc projection)...")
    cnn = CNNAutoencoder(loss_fn=FrobeniusFidelityLoss()).to(device)
    cnn.load_state_dict(torch.load(
        f"{base_dir}/checkpoints_12/cnn/best.pt",
        map_location=device
    ))
    results['cnn'] = evaluate_model(cnn, test_chunks, device, "CNN", project_output=True)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Model':<25} {'Fidelity':>12} {'Improvement':>12}")
    print("-"*50)
    baseline = results['mlp_cholesky'][2]
    print(f"{'Baseline (noisy)':<25} {baseline:>12.4f} {1.0:>12.2f}x")
    for name, (fid, std, _) in results.items():
        print(f"{name:<25} {fid:>12.4f} {fid/baseline:>12.2f}x")


if __name__ == "__main__":
    main()
