"""Generate Uhlmann fidelity comparison bar chart for Architecture Comparison section."""
import matplotlib.pyplot as plt
import numpy as np

# Final Uhlmann fidelity values from eval_models_on_uhlmann.py
# Using capacity-matched transformer (751k) for fair comparison
RESULTS = {
    "CNN\nFrobenius": 0.30,
    "CNN\nPhysics": 0.06,
    "Transformer (751k)\nFrobenius": 0.87,
    "Transformer (751k)\nPhysics": 0.26,
}


def main():
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    labels = list(RESULTS.keys())
    values = list(RESULTS.values())
    colors = ["#1f77b4", "#1f77b4", "#2ca02c", "#2ca02c"]
    hatches = ["", "//", "", "//"]

    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=1)

    # Add hatching to physics models
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    ax.set_ylabel("Uhlmann Fidelity", fontsize=12)
    ax.set_xlabel("Model Configuration", fontsize=12)
    ax.set_title("Uhlmann Fidelity by Architecture and Loss Function", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.0)

    # Add baseline reference line
    ax.axhline(y=0.11, color="gray", linestyle="--", alpha=0.7, label="Noisy baseline (0.11)")

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.legend(loc="upper right")
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/uhlmann_comparison_bar.pdf")
    plt.savefig("figures/uhlmann_comparison_bar.png")
    print("Saved: figures/uhlmann_comparison_bar.pdf")
    plt.close()


if __name__ == "__main__":
    main()