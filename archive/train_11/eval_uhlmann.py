"""
Evaluate Pauli-basis models on Uhlmann fidelity.

Converts Pauli predictions back to density matrices for evaluation.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from scipy.linalg import sqrtm

from train_11.pauli_representation import density_matrix_to_pauli_basis, pauli_basis_to_density_matrix
from train_11.mlp_pauli import MLPPauliAutoencoder
from train_11.transformer_pauli import TransformerPauliAutoencoder
from train_11.pauli_loss import PauliFrobeniusLoss

from training_loop.dataset.load_chunks import load_chunks
from training_loop.dataset.split_chunks import split_chunks


def uhlmann_fidelity(rho, sigma, debug=False):
    """
    Compute Uhlmann fidelity F(rho, sigma) = (Tr[sqrt(sqrt(rho) sigma sqrt(rho))])^2
    """
    try:
        sqrt_rho = sqrtm(rho)
        inner = sqrt_rho @ sigma @ sqrt_rho
        sqrt_inner = sqrtm(inner)
        fid = np.real(np.trace(sqrt_inner)) ** 2
        if debug:
            print(f"  sqrt_rho trace: {np.trace(sqrt_rho):.4f}")
            print(f"  inner trace: {np.trace(inner):.4f}")
            print(f"  fid: {fid:.4f}")
        return np.clip(fid, 0, 1)
    except Exception as e:
        if debug:
            print(f"  Uhlmann error: {e}")
        return 0.0


def project_to_valid_density_matrix(rho):
    """Project to valid density matrix (Hermitian, PSD, trace 1)."""
    # Make Hermitian
    rho = (rho + rho.conj().T) / 2

    # Project to PSD
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.maximum(eigvals, 0)
    rho = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T

    # Normalize trace
    rho = rho / np.trace(rho)

    return rho


def evaluate_model(model, test_chunks, device, model_name):
    """Evaluate a Pauli model on Uhlmann fidelity."""
    model.eval()

    fidelities = []
    baseline_fidelities = []
    first_sample = True

    with torch.no_grad():
        for chunk_idx, (X, Y) in enumerate(test_chunks):
            # X, Y are (N, 2, 32, 32) density matrices
            batch_size = X.shape[0]

            for i in range(batch_size):
                # Get noisy and clean density matrices
                # X[i] is (32, 32, 2) where last dim is [real, imag]
                X_i = X[i].numpy() if hasattr(X[i], 'numpy') else np.array(X[i])
                Y_i = Y[i].numpy() if hasattr(Y[i], 'numpy') else np.array(Y[i])

                noisy_dm = X_i[:, :, 0] + 1j * X_i[:, :, 1]  # (32, 32) complex
                clean_dm = Y_i[:, :, 0] + 1j * Y_i[:, :, 1]  # (32, 32) complex

                # Convert to Pauli basis
                noisy_pauli = density_matrix_to_pauli_basis(X[i], n_qubits=5)
                noisy_pauli_tensor = torch.tensor(noisy_pauli, dtype=torch.float32, device=device).unsqueeze(0)

                # Get model prediction
                pred_pauli = model(noisy_pauli_tensor).cpu().numpy()[0]

                # Convert back to density matrix
                pred_dm = pauli_basis_to_density_matrix(pred_pauli, n_qubits=5)
                pred_dm = project_to_valid_density_matrix(pred_dm)

                # Debug first sample
                if first_sample:
                    print(f"\n  DEBUG first sample:")
                    print(f"    X[i] shape: {X[i].shape}, Y[i] shape: {Y[i].shape}")
                    print(f"    noisy_dm shape: {noisy_dm.shape}, clean_dm shape: {clean_dm.shape}")
                    print(f"    noisy_dm trace: {np.trace(noisy_dm):.4f}")
                    print(f"    clean_dm trace: {np.trace(clean_dm):.4f}")
                    print(f"    pred_dm trace: {np.trace(pred_dm):.4f}")
                    print(f"    noisy_pauli range: [{noisy_pauli.min():.4f}, {noisy_pauli.max():.4f}]")
                    print(f"    pred_pauli range: [{pred_pauli.min():.4f}, {pred_pauli.max():.4f}]")
                    print(f"    pred_dm shape: {pred_dm.shape}, max abs: {np.abs(pred_dm).max():.4f}")

                # Compute fidelities
                model_fid = uhlmann_fidelity(pred_dm, clean_dm, debug=first_sample)
                baseline_fid = uhlmann_fidelity(noisy_dm, clean_dm, debug=first_sample)

                if first_sample:
                    print(f"    model_fid: {model_fid:.4f}")
                    print(f"    baseline_fid: {baseline_fid:.4f}")
                    first_sample = False

                fidelities.append(model_fid)
                baseline_fidelities.append(baseline_fid)

            if (chunk_idx + 1) % 10 == 0:
                print(f"  Processed {chunk_idx + 1}/{len(test_chunks)} chunks...")

    fidelities = np.array(fidelities)
    baseline_fidelities = np.array(baseline_fidelities)

    print(f"\n{model_name} Results:")
    print(f"  Model Uhlmann Fidelity: {fidelities.mean():.4f} ± {fidelities.std():.4f}")
    print(f"  Baseline (noisy) Fidelity: {baseline_fidelities.mean():.4f} ± {baseline_fidelities.std():.4f}")
    print(f"  Improvement: {fidelities.mean() / baseline_fidelities.mean():.2f}x")

    return fidelities, baseline_fidelities


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test data
    print("Loading dataset...")
    chunks = load_chunks("dataset_smaller")
    _, _, test_chunks = split_chunks(chunks)
    print(f"Test chunks: {len(test_chunks)}")

    # Evaluate MLP
    print("\n" + "="*60)
    print("Evaluating MLP Pauli model...")
    print("="*60)

    mlp = MLPPauliAutoencoder(loss_fn=PauliFrobeniusLoss()).to(device)
    mlp_ckpt = "train_11/checkpoints_11/mlp_pauli_frob/best.pt"
    if os.path.exists(mlp_ckpt):
        mlp.load_state_dict(torch.load(mlp_ckpt, map_location=device))
        mlp_fids, mlp_baseline = evaluate_model(mlp, test_chunks, device, "MLP Pauli")
    else:
        print(f"Checkpoint not found: {mlp_ckpt}")

    # Evaluate Transformer
    print("\n" + "="*60)
    print("Evaluating Transformer Pauli model...")
    print("="*60)

    transformer = TransformerPauliAutoencoder(loss_fn=PauliFrobeniusLoss()).to(device)
    transformer_ckpt = "train_11/checkpoints_11/transformer_pauli_frob/best.pt"
    if os.path.exists(transformer_ckpt):
        transformer.load_state_dict(torch.load(transformer_ckpt, map_location=device))
        trans_fids, trans_baseline = evaluate_model(transformer, test_chunks, device, "Transformer Pauli")
    else:
        print(f"Checkpoint not found: {transformer_ckpt}")

    print("\n" + "="*60)
    print("Summary")
    print("="*60)


if __name__ == "__main__":
    main()
