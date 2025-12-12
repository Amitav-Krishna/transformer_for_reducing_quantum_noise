"""
Generate baseline comparison charts showing:
1. Ground truth fidelity (noisy vs clean) by noise type/level
2. Model improvement over baseline (delta fidelity)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Paths
CSVS_DIR = "../csvs_2"
NOISE_CELLS_DIR = f"{CSVS_DIR}/noise_cells"
GT_PATH = f"{CSVS_DIR}/uhlmann_ground_truth/noisy_vs_clean_test_uhlmann.csv"
OUTPUT_DIR = "."

NOISE_TYPES = ["depolarizing", "amplitude_damping", "phase_damping", "bitflip", "mixed"]
NOISE_LEVELS = [0.05, 0.10, 0.15, 0.20]

MODELS = {
    "CNN Frobenius": "cnn_frob_noise_cells.csv",
    "CNN Physics": "cnn_physics_noise_cells.csv",
    "Transformer Frobenius": "transformer_frob_noise_cells.csv",
    "Transformer Physics": "transformer_physics_noise_cells.csv",
}


def load_ground_truth():
    """Load and aggregate ground truth (noisy vs clean) fidelities."""
    gt = pd.read_csv(GT_PATH)
    summary = gt.groupby(["noise_type", "noise_level"])["uhlmann_fidelity"].agg(
        ["mean", "std", "count"]
    ).reset_index()
    summary.columns = ["noise_type", "noise_level", "mean_fidelity", "std_fidelity", "count"]

    # Save for reference
    os.makedirs(NOISE_CELLS_DIR, exist_ok=True)
    summary.to_csv(f"{NOISE_CELLS_DIR}/ground_truth_noise_cells.csv", index=False)

    return summary


def load_model_data(filename):
    """Load model noise cell data."""
    return pd.read_csv(f"{NOISE_CELLS_DIR}/{filename}")


def make_baseline_heatmap(gt_df):
    """Create heatmap showing baseline (noisy vs clean) fidelity."""
    mat = np.zeros((len(NOISE_TYPES), len(NOISE_LEVELS)))

    for i, ntype in enumerate(NOISE_TYPES):
        for j, nlevel in enumerate(NOISE_LEVELS):
            row = gt_df[(gt_df["noise_type"] == ntype) & (gt_df["noise_level"] == nlevel)]
            mat[i, j] = row["mean_fidelity"].iloc[0] if len(row) > 0 else np.nan

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(NOISE_LEVELS)))
    ax.set_yticks(np.arange(len(NOISE_TYPES)))
    ax.set_xticklabels(NOISE_LEVELS)
    ax.set_yticklabels(NOISE_TYPES)
    ax.set_xlabel("Noise Level", fontsize=12)
    ax.set_ylabel("Noise Type", fontsize=12)
    ax.set_title("Baseline: Noisy vs Clean Uhlmann Fidelity", fontsize=14, fontweight="bold")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="white", fontsize=8)

    fig.colorbar(im, ax=ax, label="Uhlmann Fidelity")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/baseline_noisy_vs_clean_heatmap.pdf")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/baseline_noisy_vs_clean_heatmap.pdf")


def make_improvement_heatmaps(gt_df):
    """Create heatmaps showing fidelity improvement (model - baseline) for each model."""
    # Build baseline matrix
    baseline_mat = np.zeros((len(NOISE_TYPES), len(NOISE_LEVELS)))
    for i, ntype in enumerate(NOISE_TYPES):
        for j, nlevel in enumerate(NOISE_LEVELS):
            row = gt_df[(gt_df["noise_type"] == ntype) & (gt_df["noise_level"] == nlevel)]
            baseline_mat[i, j] = row["mean_fidelity"].iloc[0] if len(row) > 0 else np.nan

    for model_name, filename in MODELS.items():
        model_df = load_model_data(filename)

        model_mat = np.zeros((len(NOISE_TYPES), len(NOISE_LEVELS)))
        for i, ntype in enumerate(NOISE_TYPES):
            for j, nlevel in enumerate(NOISE_LEVELS):
                row = model_df[(model_df["noise_type"] == ntype) & (model_df["noise_level"] == nlevel)]
                model_mat[i, j] = row["mean_fidelity"].iloc[0] if len(row) > 0 else np.nan

        # Improvement = model fidelity - baseline fidelity
        delta_mat = model_mat - baseline_mat

        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        # Use diverging colormap centered at 0
        vmax = max(abs(np.nanmin(delta_mat)), abs(np.nanmax(delta_mat)), 0.5)
        im = ax.imshow(delta_mat, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)

        ax.set_xticks(np.arange(len(NOISE_LEVELS)))
        ax.set_yticks(np.arange(len(NOISE_TYPES)))
        ax.set_xticklabels(NOISE_LEVELS)
        ax.set_yticklabels(NOISE_TYPES)
        ax.set_xlabel("Noise Level", fontsize=12)
        ax.set_ylabel("Noise Type", fontsize=12)
        ax.set_title(f"{model_name}: Fidelity Improvement over Baseline", fontsize=14, fontweight="bold")

        for i in range(delta_mat.shape[0]):
            for j in range(delta_mat.shape[1]):
                val = delta_mat[i, j]
                if not np.isnan(val):
                    color = "black" if abs(val) < vmax * 0.5 else "white"
                    ax.text(j, i, f"{val:+.3f}", ha="center", va="center", color=color, fontsize=8)

        fig.colorbar(im, ax=ax, label="Δ Fidelity (Model - Baseline)")
        plt.tight_layout()

        out_name = filename.replace("_noise_cells.csv", "_improvement_heatmap.pdf")
        plt.savefig(f"{OUTPUT_DIR}/{out_name}")
        plt.close()
        print(f"Saved: {OUTPUT_DIR}/{out_name}")


def make_combined_bar_chart(gt_df):
    """Create grouped bar chart comparing baseline vs all models (aggregated)."""
    # Overall baseline
    gt_raw = pd.read_csv(GT_PATH)
    baseline_mean = gt_raw["uhlmann_fidelity"].mean()
    baseline_std = gt_raw["uhlmann_fidelity"].std()

    # Load overall model stats
    model_stats = pd.read_csv(f"{CSVS_DIR}/uhlmann_fidelity.csv")

    labels = ["Baseline\n(Noisy)"] + [m.replace(" ", "\n") for m in MODELS.keys()]
    means = [baseline_mean] + [
        model_stats[model_stats["model"] == f.replace("_noise_cells.csv", "")]["mean_fidelity"].iloc[0]
        for f in MODELS.values()
    ]
    stds = [baseline_std] + [
        model_stats[model_stats["model"] == f.replace("_noise_cells.csv", "")]["std_fidelity"].iloc[0]
        for f in MODELS.values()
    ]

    colors = ["#808080", "#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="black", linewidth=1)

    ax.set_ylabel("Uhlmann Fidelity", fontsize=12)
    ax.set_title("Reconstruction Fidelity: Baseline vs Models", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=baseline_mean, color="gray", linestyle="--", alpha=0.5, label="Baseline")

    # Annotate bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{mean:.2f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/baseline_vs_models_bar.pdf")
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/baseline_vs_models_bar.pdf")


def make_improvement_summary_table(gt_df):
    """Print summary table of improvements."""
    gt_raw = pd.read_csv(GT_PATH)
    baseline_mean = gt_raw["uhlmann_fidelity"].mean()

    model_stats = pd.read_csv(f"{CSVS_DIR}/uhlmann_fidelity.csv")

    print("\n" + "="*70)
    print("IMPROVEMENT SUMMARY")
    print("="*70)
    print(f"{'Model':<25} {'Fidelity':>12} {'Baseline':>12} {'Improvement':>15}")
    print("-"*70)
    print(f"{'Baseline (Noisy)':<25} {baseline_mean:>12.4f} {'-':>12} {'-':>15}")

    for name, filename in MODELS.items():
        model_key = filename.replace("_noise_cells.csv", "")
        row = model_stats[model_stats["model"] == model_key]
        if len(row) > 0:
            fid = row["mean_fidelity"].iloc[0]
            delta = fid - baseline_mean
            print(f"{name:<25} {fid:>12.4f} {baseline_mean:>12.4f} {delta:>+15.4f}")
    print("="*70)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("Loading ground truth data...")
    gt_df = load_ground_truth()

    print("\nGenerating baseline heatmap...")
    make_baseline_heatmap(gt_df)

    print("\nGenerating improvement heatmaps...")
    make_improvement_heatmaps(gt_df)

    print("\nGenerating combined bar chart...")
    make_combined_bar_chart(gt_df)

    make_improvement_summary_table(gt_df)

    print("\nDone!")
