"""
Bar chart comparing CNN Frobenius vs Transformer Frobenius Uhlmann fidelity.
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from csvs_2/uhlmann_fidelity.csv and ground truth
models = ["Baseline\n(Noisy)", "CNN\nFrobenius", "Transformer\nFrobenius"]
means = [0.1114, 0.3048, 0.9495]
stds = [0.1500, 0.2843, 0.1242]

colors = ["#808080", "#1f77b4", "#2ca02c"]

fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

x = np.arange(len(models))
bars = ax.bar(x, means, yerr=stds, capsize=8, color=colors, edgecolor="black", linewidth=1.5)

ax.set_ylabel("Uhlmann Fidelity", fontsize=14)
ax.set_title("Uhlmann Fidelity (Higher is Better)", fontsize=16, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.set_ylim(0, 1.15)

# Annotate bars
for bar, mean, std in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.03,
            f"{mean:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("figures/uhlmann_cnn_vs_transformer.pdf")
plt.savefig("figures/uhlmann_cnn_vs_transformer.png")
print("Saved uhlmann_cnn_vs_transformer.pdf and .png")
