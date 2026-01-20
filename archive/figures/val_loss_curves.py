#!/usr/bin/env python3
"""
Generate validation loss curves for paper_v4.

Compares 5-qubit vs 8-qubit training dynamics for both Transformer and MLP.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Use a clean style
plt.style.use("seaborn-v0_8-whitegrid")


def load_val_loss(model_name):
    """Load validation loss from CSV."""
    path = f"train_16/csvs_16/{model_name}_val.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find {path}")
    df = pd.read_csv(path)
    return df["epoch"].values, df["loss"].values


def main():
    os.makedirs("figures", exist_ok=True)

    # Load data
    models = {
        "transformer_5qubit": ("Transformer 5q", "tab:blue", "-"),
        "mlp_5qubit": ("MLP 5q", "tab:orange", "-"),
        "transformer_8qubit": ("Transformer 8q", "tab:blue", "--"),
        "mlp_8qubit": ("MLP 8q", "tab:orange", "--"),
    }

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    for model_key, (label, color, linestyle) in models.items():
        try:
            epochs, losses = load_val_loss(model_key)
            ax.plot(
                epochs,
                losses,
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=2,
            )
        except FileNotFoundError as e:
            print(f"Warning: {e}")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation Loss (Frobenius)", fontsize=12)
    ax.set_title("Training Dynamics: 5-Qubit vs 8-Qubit", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)

    # Set y-axis to start from 0 for better comparison
    ax.set_ylim(0, 1.0)
    ax.set_xlim(1, 100)

    # Add annotation for the gap
    ax.annotate(
        "8-qubit models\nfail to learn",
        xy=(50, 0.91),
        xytext=(70, 0.75),
        fontsize=10,
        ha="center",
        arrowprops=dict(arrowstyle="->", color="gray"),
    )

    ax.annotate(
        "5-qubit models\nlearn effectively",
        xy=(50, 0.45),
        xytext=(30, 0.25),
        fontsize=10,
        ha="center",
        arrowprops=dict(arrowstyle="->", color="gray"),
    )

    plt.tight_layout()
    plt.savefig("figures/val_loss_curves.pdf", bbox_inches="tight")
    plt.savefig("figures/val_loss_curves.png", bbox_inches="tight")
    print("Saved: figures/val_loss_curves.pdf")
    print("Saved: figures/val_loss_curves.png")


if __name__ == "__main__":
    main()
