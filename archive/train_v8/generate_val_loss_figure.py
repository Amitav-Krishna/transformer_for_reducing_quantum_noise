#!/usr/bin/env python3
"""Generate validation loss overlay figure (MLP vs Transformer) for v8."""

import pandas as pd
import matplotlib.pyplot as plt
import os

# Ensure we're in repo root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(REPO_ROOT)

os.makedirs("figures", exist_ok=True)


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


# Load data from v8
mlp_val = load_val_curve("train_v8/csvs_8/mlp.csv")
transformer_val = load_val_curve("train_v8/csvs_8/transformer.csv")

fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
fig.patch.set_facecolor('white')

ax.plot(mlp_val["epoch"], mlp_val["loss"], linewidth=2, label="MLP (residual)", color="#d62728")
ax.plot(transformer_val["epoch"], transformer_val["loss"], linewidth=2, label="Transformer", color="#1f77b4")

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Validation Loss", fontsize=12)
ax.set_title("Validation Loss vs Epoch", fontsize=14, fontweight="bold")
ax.legend(loc="upper right")
ax.tick_params(labelsize=10)
ax.set_ylim(0, max(mlp_val["loss"].max(), transformer_val["loss"].max()) * 1.05)

plt.tight_layout()
plt.savefig("figures/mlp_transformer_val_loss.pdf")
plt.savefig("figures/mlp_transformer_val_loss.png")
plt.close()
print("Generated: figures/mlp_transformer_val_loss.pdf")
