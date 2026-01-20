import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
def frobenius_fidelity(rho, sigma, eps=1e-8):
    """
    Normalized Frobenius inner product (cosine similarity for matrices).
    rho, sigma: (B, H, W, 2) real/imag channels in last dim
    returns (B,) fidelities in [-1, 1], typically [0, 1] for similar matrices
    """
    a = rho[..., 0] + 1j * rho[..., 1]
    b = sigma[..., 0] + 1j * sigma[..., 1]

    num = torch.real(torch.sum(a.conj() * b, dim=(-1, -2)))
    denom = torch.sqrt(
        torch.sum(torch.abs(a)**2, dim=(-1, -2)) *
        torch.sum(torch.abs(b)**2, dim=(-1, -2)) + eps
    )

    fid = torch.clamp(num / denom, -1, 1)
    return fid


def main():
    print("Loading dataset...")
    chunks = load_chunks_with_metadata("dataset_smaller")
    _, _, test_chunks = split_chunks(chunks, seed=42)

    all_uhlmann = []
    all_frobenius = []

    print("Computing fidelities for noisy vs clean test states...")
    for blob in test_chunks:
        X = blob["X"]  # noisy
        Y = blob["Y"]  # clean
        meta = blob["meta"]

        X = X.detach().clone().to(dtype=torch.float64)
        Y = Y.detach().clone().to(dtype=torch.float64)

        uhl_fids = uhlmann_fidelity(X, Y).cpu().numpy()
        frob_fids = frobenius_fidelity(X, Y).cpu().numpy()

        for uf, ff, m in zip(uhl_fids, frob_fids, meta):
            all_uhlmann.append({
                "noise_type": m["noise_type"],
                "noise_level": m["noise_level"],
                "uhlmann_fidelity": float(uf)
            })
            all_frobenius.append({
                "noise_type": m["noise_type"],
                "noise_level": m["noise_level"],
                "frobenius_fidelity": float(ff)
            })

    df_uhl = pd.DataFrame(all_uhlmann)
    df_frob = pd.DataFrame(all_frobenius)

    # Aggregate by noise type and level
    uhl_agg = df_uhl.groupby(["noise_type", "noise_level"])["uhlmann_fidelity"].agg(["mean", "std"]).reset_index()
    frob_agg = df_frob.groupby(["noise_type", "noise_level"])["frobenius_fidelity"].agg(["mean", "std"]).reset_index()

    # Merge
    merged = uhl_agg.merge(frob_agg, on=["noise_type", "noise_level"], suffixes=("_uhlmann", "_frobenius"))

    # Print summary
    print("\n" + "="*80)
    print("COMPARISON: Uhlmann vs Frobenius Fidelity (Noisy vs Clean Baseline)")
    print("="*80)
    print(f"{'Noise Type':<20} {'Level':>6} {'Uhlmann':>12} {'Frobenius':>12}")
    print("-"*60)
    for _, row in merged.iterrows():
        print(f"{row['noise_type']:<20} {row['noise_level']:>6.2f} {row['mean_uhlmann']:>12.4f} {row['mean_frobenius']:>12.4f}")

    overall_uhl = df_uhl["uhlmann_fidelity"].mean()
    overall_frob = df_frob["frobenius_fidelity"].mean()
    print("-"*60)
    print(f"{'OVERALL MEAN':<20} {'':>6} {overall_uhl:>12.4f} {overall_frob:>12.4f}")

    # Create grouped bar chart
    noise_types = ["depolarizing", "amplitude_damping", "phase_damping", "bitflip", "mixed"]
    noise_levels = [0.05, 0.10, 0.15, 0.20]

    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)

    x_labels = []
    uhl_vals = []
    frob_vals = []

    for ntype in noise_types:
        for nlevel in noise_levels:
            row = merged[(merged["noise_type"] == ntype) & (merged["noise_level"] == nlevel)]
            if len(row) > 0:
                x_labels.append(f"{ntype[:4]}\n{nlevel}")
                uhl_vals.append(row["mean_uhlmann"].iloc[0])
                frob_vals.append(row["mean_frobenius"].iloc[0])

    x = np.arange(len(x_labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, uhl_vals, width, label="Uhlmann Fidelity", color="#2ca02c", alpha=0.8)
    bars2 = ax.bar(x + width/2, frob_vals, width, label="Frobenius Fidelity", color="#1f77b4", alpha=0.8)

    ax.set_ylabel("Fidelity", fontsize=12)
    ax.set_xlabel("Noise Type / Level", fontsize=12)
    ax.set_title("Baseline Fidelity: Noisy Input vs Clean Target\n(Uhlmann vs Frobenius)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.0)
    ax.axhline(y=overall_uhl, color="#2ca02c", linestyle="--", alpha=0.5, label=f"Uhlmann mean: {overall_uhl:.2f}")
    ax.axhline(y=overall_frob, color="#1f77b4", linestyle="--", alpha=0.5, label=f"Frobenius mean: {overall_frob:.2f}")

    # Add vertical separators between noise types
    for i in range(1, 5):
        ax.axvline(x=i*4 - 0.5, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    out_path = "figures/uhlmann_vs_frobenius_baseline.pdf"
    plt.savefig(out_path)
    plt.savefig(out_path.replace(".pdf", ".png"))
    print(f"\nSaved: {out_path}")
    plt.close()

    # Also create a heatmap comparison (side by side)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    for ax_idx, (metric, title, cmap) in enumerate([
        ("mean_uhlmann", "Uhlmann Fidelity", "Greens"),
        ("mean_frobenius", "Frobenius Fidelity", "Blues")
    ]):
        ax = axes[ax_idx]
        mat = np.zeros((len(noise_types), len(noise_levels)))

        for i, ntype in enumerate(noise_types):
            for j, nlevel in enumerate(noise_levels):
                row = merged[(merged["noise_type"] == ntype) & (merged["noise_level"] == nlevel)]
                if len(row) > 0:
                    mat[i, j] = row[metric].iloc[0]
                else:
                    mat[i, j] = np.nan

        im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(noise_levels)))
        ax.set_yticks(np.arange(len(noise_types)))
        ax.set_xticklabels(noise_levels)
        ax.set_yticklabels(noise_types)
        ax.set_xlabel("Noise Level")
        ax.set_ylabel("Noise Type")
        ax.set_title(title, fontweight="bold")

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            color="white" if val > 0.5 else "black", fontsize=9)

        fig.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle("Baseline: Noisy Input vs Clean Target", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path2 = "figures/uhlmann_vs_frobenius_heatmaps.pdf"
    plt.savefig(out_path2)
    plt.savefig(out_path2.replace(".pdf", ".png"))
    print(f"Saved: {out_path2}")
    plt.close()


if __name__ == "__main__":
    main()