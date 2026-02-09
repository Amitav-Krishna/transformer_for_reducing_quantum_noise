#!/usr/bin/env python3
"""
Two-panel validation loss comparison: unnormalized vs normalized
for all four model/qubit combinations.
"""

import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

BASE = "../known_good/code"

# train_16: unnormalized (epoch-level)
UNNORM = {
    "5q MLP":  f"{BASE}/train_16/csvs_16/mlp_5qubit_val.csv",
    "5q Txf":  f"{BASE}/train_16/csvs_16/transformer_5qubit_val.csv",
    "8q MLP":  f"{BASE}/train_16/csvs_16/mlp_8qubit_val.csv",
    "8q Txf":  f"{BASE}/train_16/csvs_16/transformer_8qubit_val.csv",
}

# train_17: normalized (batch-level, extract epoch val_loss)
NORM = {
    "5q MLP":  f"{BASE}/train_17/results/5q_mlp/mlp.csv",
    "5q Txf":  f"{BASE}/train_17/results/5q_transformer/transformer.csv",
    "8q MLP":  f"{BASE}/train_17/results/8q_mlp/mlp_8q.csv",
    "8q Txf":  f"{BASE}/train_17/results/8q_transformer/transformer_8q.csv",
}


def load_unnorm(path):
    df = pd.read_csv(path)
    return df["epoch"].values, df["loss"].values


def load_norm(path):
    df = pd.read_csv(path)
    val = df.dropna(subset=["val_loss"])
    return val["epoch"].values, val["val_loss"].values


STYLES = {
    "MLP": {"color": "tab:orange", "linewidth": 2},
    "Txf": {"color": "tab:blue", "linewidth": 2},
}


def plot_panel(ax, qubit_label, models):
    """Plot unnormalized (dashed) and normalized (solid) for two models."""
    for model_key in models:
        short = model_key.split()[-1]  # "MLP" or "Txf"
        style = STYLES[short]

        # Unnormalized
        epochs, loss = load_unnorm(UNNORM[model_key])
        ax.plot(epochs, loss, linestyle="--", alpha=0.7,
                label=f"{short} (unnorm)", **style)

        # Normalized
        epochs, loss = load_norm(NORM[model_key])
        ax.plot(epochs, loss, linestyle="-",
                label=f"{short} (norm)", **style)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Validation Loss", fontsize=11)
    ax.set_xlim(1, 100)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(qubit_label, fontsize=12)


def save_panel(filename, qubit_label, models, ylim):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    plot_panel(ax, qubit_label, models)
    ax.set_ylim(*ylim)
    plt.tight_layout()
    plt.savefig(f"figures/{filename}.pdf", bbox_inches="tight")
    plt.savefig(f"figures/{filename}.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: figures/{filename}.pdf")


def main():
    save_panel("val_loss_5qubit", "5-Qubit Models",
               ["5q MLP", "5q Txf"], (0.0, 1.0))
    save_panel("val_loss_8qubit", "8-Qubit Models",
               ["8q MLP", "8q Txf"], (0.84, 0.95))


if __name__ == "__main__":
    main()
