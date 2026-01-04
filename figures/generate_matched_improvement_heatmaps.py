"""Generate improvement heatmaps for the capacity-matched transformer (751k)."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

INPUT_DIR = "csvs_2/noise_cells"

# Fixed ordering for consistent heatmaps
NOISE_TYPES = ["depolarizing", "amplitude_damping", "phase_damping", "bitflip", "mixed"]
NOISE_LEVELS = [0.05, 0.10, 0.15, 0.20]

# Baseline fidelity values (noisy vs clean)
BASELINE = {
    ("depolarizing", 0.05): 0.09, ("depolarizing", 0.10): 0.04, ("depolarizing", 0.15): 0.03, ("depolarizing", 0.20): 0.03,
    ("amplitude_damping", 0.05): 0.20, ("amplitude_damping", 0.10): 0.07, ("amplitude_damping", 0.15): 0.04, ("amplitude_damping", 0.20): 0.03,
    ("phase_damping", 0.05): 0.55, ("phase_damping", 0.10): 0.33, ("phase_damping", 0.15): 0.24, ("phase_damping", 0.20): 0.19,
    ("bitflip", 0.05): 0.08, ("bitflip", 0.10): 0.04, ("bitflip", 0.15): 0.03, ("bitflip", 0.20): 0.03,
    ("mixed", 0.05): 0.14, ("mixed", 0.10): 0.05, ("mixed", 0.15): 0.04, ("mixed", 0.20): 0.03,
}

MODELS = {
    "Transformer-Matched Frobenius": "transformer_matched_frob_noise_cells.csv",
    "Transformer-Matched Physics": "transformer_matched_physics_noise_cells.csv",
}


def make_improvement_heatmap(df, model_name, out_path):
    """Create improvement heatmap (model fidelity - baseline)."""
    mat = np.zeros((len(NOISE_TYPES), len(NOISE_LEVELS)))

    for i, ntype in enumerate(NOISE_TYPES):
        for j, nlevel in enumerate(NOISE_LEVELS):
            row = df[(df["noise_type"] == ntype) & (df["noise_level"] == nlevel)]
            baseline = BASELINE.get((ntype, nlevel), 0.11)
            if len(row) > 0:
                mat[i, j] = row["mean_fidelity"].iloc[0] - baseline
            else:
                mat[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    # Use diverging colormap centered at 0
    vmax = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)))
    im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(len(NOISE_LEVELS)))
    ax.set_yticks(np.arange(len(NOISE_TYPES)))
    ax.set_xticklabels(NOISE_LEVELS)
    ax.set_yticklabels(NOISE_TYPES)

    ax.set_xlabel("Noise Level", fontsize=12)
    ax.set_ylabel("Noise Type", fontsize=12)
    ax.set_title(f"{model_name}: Improvement over Baseline", fontsize=14, fontweight="bold")

    # Annotate cells
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > vmax * 0.5 else "black"
                sign = "+" if val > 0 else ""
                ax.text(j, i, f"{sign}{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(im, ax=ax, label="Fidelity Improvement")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved: {out_path}")
    plt.close()


def main():
    os.makedirs("figures", exist_ok=True)

    for model_name, filename in MODELS.items():
        df = pd.read_csv(f"{INPUT_DIR}/{filename}")
        out_name = filename.replace("_noise_cells.csv", "_improvement_heatmap.pdf")
        make_improvement_heatmap(df, model_name, f"figures/{out_name}")


if __name__ == "__main__":
    main()