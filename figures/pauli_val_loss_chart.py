#!/usr/bin/env python3
"""
Create validation loss chart for Pauli models and verify rank correlation.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

def load_val_curve(path):
    """Extract validation-loss curve from CSV."""
    df = pd.read_csv(path)
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df = df.dropna(subset=["epoch"])
    df["epoch"] = df["epoch"].astype(int)
    df_val = df[df["chunk_id"] == "val"].copy()
    df_val = df_val.rename(columns={"val_loss": "loss"})
    df_val = df_val.sort_values("epoch")
    return df_val

# Load data
mlp_val = load_val_curve("train_11/csvs_11/mlp_pauli_frob.csv")
transformer_val = load_val_curve("train_11/csvs_11/transformer_pauli_frob.csv")

print("MLP Pauli validation loss:")
print(f"  Initial: {mlp_val['loss'].iloc[0]:.4f}")
print(f"  Final: {mlp_val['loss'].iloc[-1]:.4f}")
print(f"  Best: {mlp_val['loss'].min():.4f}")
print(f"  Epochs: {len(mlp_val)}")

print("\nTransformer Pauli validation loss:")
print(f"  Initial: {transformer_val['loss'].iloc[0]:.4f}")
print(f"  Final: {transformer_val['loss'].iloc[-1]:.4f}")
print(f"  Best: {transformer_val['loss'].min():.4f}")
print(f"  Epochs: {len(transformer_val)}")

# Create figure
fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
fig.patch.set_facecolor('white')

ax.plot(mlp_val["epoch"], mlp_val["loss"], linewidth=2, label="MLP Pauli", color="#d62728")
ax.plot(transformer_val["epoch"], transformer_val["loss"], linewidth=2, label="Transformer Pauli", color="#1f77b4")

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Validation Loss (Pauli Frobenius)", fontsize=12)
ax.set_title("Pauli Basis Models: Validation Loss vs Epoch", fontsize=14, fontweight="bold")
ax.legend(loc="upper right")
ax.tick_params(labelsize=10)
ax.set_ylim(0, max(mlp_val["loss"].max(), transformer_val["loss"].max()) * 1.05)

plt.tight_layout()
plt.savefig("figures/pauli_val_loss.pdf")
plt.savefig("figures/pauli_val_loss.png")
plt.close()
print("\nSaved: figures/pauli_val_loss.pdf")

# Now compute rank correlation between Pauli Frobenius loss and Uhlmann fidelity
# We need to sample predictions at different loss values and compute Uhlmann fidelity
print("\n" + "="*60)
print("Verifying rank correlation: Pauli Frobenius vs Uhlmann Fidelity")
print("="*60)

import torch
from scipy.linalg import sqrtm
from train_11.transformer_pauli import TransformerPauliAutoencoder
from train_11.pauli_representation import density_matrix_to_pauli_basis, pauli_basis_to_density_matrix
from train_11.pauli_loss import PauliFrobeniusLoss
from training_loop.dataset.load_chunks import load_chunks
from training_loop.dataset.split_chunks import split_chunks

device = torch.device('cpu')

def uhlmann_fidelity(rho, sigma):
    """Compute Uhlmann fidelity."""
    try:
        sqrt_rho = sqrtm(rho)
        inner = sqrt_rho @ sigma @ sqrt_rho
        sqrt_inner = sqrtm(inner)
        fid = np.real(np.trace(sqrt_inner)) ** 2
        return np.clip(fid, 0, 1)
    except:
        return 0.0

def project_to_valid_dm(rho):
    """Project to valid density matrix."""
    rho = (rho + rho.conj().T) / 2
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.maximum(eigvals, 0)
    rho = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
    rho = rho / np.trace(rho)
    return rho

# Load model and test data
model = TransformerPauliAutoencoder(loss_fn=PauliFrobeniusLoss())
model.load_state_dict(torch.load('train_11/checkpoints_11/transformer_pauli_frob/best.pt', map_location=device))
model.eval()

print("Loading test data...")
chunks = load_chunks("dataset_smaller")
_, _, test_chunks = split_chunks(chunks)

# Sample from test set
n_samples = 200
pauli_losses = []
uhlmann_fids = []

print(f"Computing correlation on {n_samples} samples...")
sample_count = 0

with torch.no_grad():
    for chunk_idx, (X, Y) in enumerate(test_chunks):
        if sample_count >= n_samples:
            break

        for i in range(len(X)):
            if sample_count >= n_samples:
                break

            # Get density matrices
            X_i = X[i].numpy() if hasattr(X[i], 'numpy') else np.array(X[i])
            Y_i = Y[i].numpy() if hasattr(Y[i], 'numpy') else np.array(Y[i])

            # Clean density matrix
            clean_dm = Y_i[:, :, 0] + 1j * Y_i[:, :, 1]

            # Convert to Pauli and get prediction
            noisy_pauli = density_matrix_to_pauli_basis(X[i], n_qubits=5)
            noisy_pauli_tensor = torch.tensor(noisy_pauli, dtype=torch.float32).unsqueeze(0)

            pred_pauli = model(noisy_pauli_tensor).numpy()[0]
            clean_pauli = density_matrix_to_pauli_basis(Y[i], n_qubits=5)

            # Compute Pauli Frobenius loss (1 - cosine similarity)
            pred_norm = np.linalg.norm(pred_pauli)
            clean_norm = np.linalg.norm(clean_pauli)
            cosine_sim = np.dot(pred_pauli, clean_pauli) / (pred_norm * clean_norm + 1e-8)
            pauli_loss = 1 - cosine_sim

            # Convert prediction back to density matrix and compute Uhlmann
            pred_dm = pauli_basis_to_density_matrix(pred_pauli, n_qubits=5)
            pred_dm = project_to_valid_dm(pred_dm)

            uhlmann = uhlmann_fidelity(pred_dm, clean_dm)

            pauli_losses.append(pauli_loss)
            uhlmann_fids.append(uhlmann)
            sample_count += 1

            if sample_count % 50 == 0:
                print(f"  Processed {sample_count}/{n_samples} samples...")

pauli_losses = np.array(pauli_losses)
uhlmann_fids = np.array(uhlmann_fids)

# Compute Spearman correlation (loss vs fidelity should be negatively correlated)
# Or equivalently, (1-loss) vs fidelity should be positively correlated
spearman_corr, p_value = spearmanr(1 - pauli_losses, uhlmann_fids)

print(f"\n{'='*60}")
print("RESULTS: Pauli Frobenius Similarity vs Uhlmann Fidelity")
print(f"{'='*60}")
print(f"Samples: {len(pauli_losses)}")
print(f"Spearman correlation: {spearman_corr:.4f}")
print(f"P-value: {p_value:.2e}")
print(f"Pauli loss range: [{pauli_losses.min():.4f}, {pauli_losses.max():.4f}]")
print(f"Uhlmann fidelity range: [{uhlmann_fids.min():.4f}, {uhlmann_fids.max():.4f}]")

# Create scatter plot
fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
ax.scatter(1 - pauli_losses, uhlmann_fids, alpha=0.5, s=20)
ax.set_xlabel("Pauli Frobenius Similarity (1 - loss)", fontsize=12)
ax.set_ylabel("Uhlmann Fidelity", fontsize=12)
ax.set_title(f"Pauli Frobenius vs Uhlmann Fidelity\n(Spearman $\\rho$ = {spearman_corr:.3f})", fontsize=14)

# Add diagonal reference
lims = [max(ax.get_xlim()[0], ax.get_ylim()[0]), min(ax.get_xlim()[1], ax.get_ylim()[1])]
ax.plot(lims, lims, 'k--', alpha=0.3, label='y=x')
ax.legend()

plt.tight_layout()
plt.savefig("figures/pauli_frob_vs_uhlmann.pdf")
plt.savefig("figures/pauli_frob_vs_uhlmann.png")
print("\nSaved: figures/pauli_frob_vs_uhlmann.pdf")
