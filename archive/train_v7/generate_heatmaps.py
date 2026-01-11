#!/usr/bin/env python3
"""Generate per-noise-cell Uhlmann fidelity heatmaps."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure we're in repo root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(REPO_ROOT)

os.makedirs("figures", exist_ok=True)

# Fixed ordering for consistent heatmaps
NOISE_TYPES = ["depolarizing", "amplitude_damping", "phase_damping", "bitflip", "mixed"]
NOISE_LEVELS = [0.05, 0.10, 0.15, 0.20]


def make_heatmap(csv_path, model_name, out_path, vmin=0, vmax=0.25):
    """Generate heatmap from noise-cell CSV."""
    df = pd.read_csv(csv_path)

    mat = np.zeros((len(NOISE_TYPES), len(NOISE_LEVELS)))
    for i, ntype in enumerate(NOISE_TYPES):
        for j, nlevel in enumerate(NOISE_LEVELS):
            row = df[(df["noise_type"] == ntype) & (df["noise_level"] == nlevel)]
            mat[i, j] = row["mean_fidelity"].iloc[0] if len(row) > 0 else np.nan

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(NOISE_LEVELS)))
    ax.set_yticks(np.arange(len(NOISE_TYPES)))
    ax.set_xticklabels(NOISE_LEVELS)
    ax.set_yticklabels([t.replace("_", " ").title() for t in NOISE_TYPES])

    ax.set_xlabel("Noise Level", fontsize=12)
    ax.set_ylabel("Noise Type", fontsize=12)
    ax.set_title(f"{model_name}: Uhlmann Fidelity", fontsize=14, fontweight="bold")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if not np.isnan(val):
                color = "white" if val < (vmax - vmin) / 2 + vmin else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=9)

    fig.colorbar(im, ax=ax, label="Uhlmann Fidelity")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.savefig(out_path.replace(".pdf", ".png"))
    plt.close()
    return out_path


# Generate heatmaps (requires noise_cells CSVs from eval script)
INPUT_DIR = "train_v7/csvs_7/noise_cells"
models = {
    "Baseline": "baseline_noise_cells.csv",
    "MLP (residual)": "mlp_noise_cells.csv",
    "Transformer": "transformer_noise_cells.csv",
}

for model_name, filename in models.items():
    csv_path = f"{INPUT_DIR}/{filename}"
    if os.path.exists(csv_path):
        out_path = f"figures/{filename.replace('.csv', '_heatmap.pdf')}"
        make_heatmap(csv_path, model_name, out_path)
        print(f"Generated: {out_path}")
    else:
        print(f"Missing: {csv_path}")
