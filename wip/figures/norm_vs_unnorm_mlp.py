#!/usr/bin/env python3
"""
Generate unnormalized vs normalized MLP validation loss comparison.
"""

import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

UNNORM_CSV = "../known_good/code/train_16/csvs_16/mlp_5qubit_val.csv"
NORM_CSV = "../known_good/code/train_17/results/5q_mlp/mlp.csv"


def load_unnorm():
    df = pd.read_csv(UNNORM_CSV)
    return df["epoch"].values, df["loss"].values


def load_norm():
    """Extract per-epoch validation loss from the batch-level CSV."""
    df = pd.read_csv(NORM_CSV)
    val = df.dropna(subset=["val_loss"])
    return val["epoch"].values, val["val_loss"].values


def main():
    unnorm_epochs, unnorm_loss = load_unnorm()
    norm_epochs, norm_loss = load_norm()

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    ax.plot(unnorm_epochs, unnorm_loss, label="Unnormalized MLP",
            color="tab:red", linewidth=2)
    ax.plot(norm_epochs, norm_loss, label="Normalized MLP",
            color="tab:blue", linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation Loss (Frobenius)", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(1, 100)

    plt.tight_layout()
    plt.savefig("figures/norm_vs_unnorm_mlp.pdf", bbox_inches="tight")
    plt.savefig("figures/norm_vs_unnorm_mlp.png", bbox_inches="tight")
    print("Saved: figures/norm_vs_unnorm_mlp.pdf")
    print("Saved: figures/norm_vs_unnorm_mlp.png")


if __name__ == "__main__":
    main()
