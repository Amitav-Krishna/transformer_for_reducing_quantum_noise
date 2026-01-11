import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    rho, sigma: (B, H, W, 2) real/imag channels in last dim
    returns (B,) fidelities
    """
    rho_c = rho[..., 0] + 1j * rho[..., 1]
    sig_c = sigma[..., 0] + 1j * sigma[..., 1]

    sr = matrix_sqrt(rho_c)
    inner = sr @ sig_c @ sr
    s_inner = matrix_sqrt(inner)

    tr = torch.real(torch.diagonal(s_inner, dim1=-2, dim2=-1).sum(-1))
    return tr**2


@torch.no_grad()
def frobenius_similarity(rho, sigma, eps=1e-8):
    """
    Normalized Frobenius inner product (cosine similarity for matrices).
    rho, sigma: (B, H, W, 2) real/imag channels in last dim
    returns (B,) similarities in [0, 1] for similar density matrices
    """
    a = rho[..., 0] + 1j * rho[..., 1]
    b = sigma[..., 0] + 1j * sigma[..., 1]

    num = torch.real(torch.sum(a.conj() * b, dim=(-1, -2)))
    denom = torch.sqrt(
        torch.sum(torch.abs(a)**2, dim=(-1, -2)) *
        torch.sum(torch.abs(b)**2, dim=(-1, -2)) + eps
    )

    return torch.clamp(num / denom, 0, 1)


def main():
    print("Loading dataset...")
    chunks = load_chunks_with_metadata("dataset_smaller")
    _, _, test_chunks = split_chunks(chunks, seed=42)

    results = []

    print("Computing fidelities for noisy vs clean test states...")
    for blob in test_chunks:
        X = blob["X"]  # noisy
        Y = blob["Y"]  # clean
        meta = blob["meta"]

        X = X.detach().clone().to(dtype=torch.float64)
        Y = Y.detach().clone().to(dtype=torch.float64)

        uhl_fids = uhlmann_fidelity(X, Y).cpu().numpy()
        frob_sims = frobenius_similarity(X, Y).cpu().numpy()

        for uf, fs, m in zip(uhl_fids, frob_sims, meta):
            results.append({
                "noise_type": m["noise_type"],
                "noise_level": m["noise_level"],
                "uhlmann_fidelity": float(uf),
                "frobenius_similarity": float(fs)
            })

    df = pd.DataFrame(results)

    # Compute Spearman correlation
    rho, pval = spearmanr(df["frobenius_similarity"], df["uhlmann_fidelity"])
    print(f"\nSpearman correlation: ρ = {rho:.4f}, p = {pval:.2e}")

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    noise_types = ["depolarizing", "amplitude_damping", "phase_damping", "bitflip", "mixed"]
    colors = {"depolarizing": "#1f77b4", "amplitude_damping": "#ff7f0e",
              "phase_damping": "#2ca02c", "bitflip": "#d62728", "mixed": "#9467bd"}
    labels = {"depolarizing": "Depolarizing", "amplitude_damping": "Amplitude Damping",
              "phase_damping": "Phase Damping", "bitflip": "Bit Flip", "mixed": "Mixed"}

    for ntype in noise_types:
        subset = df[df["noise_type"] == ntype]
        ax.scatter(subset["frobenius_similarity"], subset["uhlmann_fidelity"],
                   c=colors[ntype], label=labels[ntype], alpha=0.5, s=15, edgecolors='none')

    ax.set_xlabel("Normalized Frobenius Similarity", fontsize=12)
    ax.set_ylabel("Uhlmann Fidelity", fontsize=12)
    ax.set_title(f"Frobenius Similarity vs Uhlmann Fidelity\n(Spearman ρ = {rho:.2f})",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Add diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='y=x')

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = "figures/uhlmann_vs_frobenius_scatter.png"
    plt.savefig(out_path, dpi=150)
    plt.savefig(out_path.replace(".png", ".pdf"))
    print(f"\nSaved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()