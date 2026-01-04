"""
Generate LaTeX table comparing element-wise and hierarchical transformers.

Output: figures/timing_comparison_table.tex
"""

import os
import pandas as pd
from pathlib import Path


def load_timing_mean(path):
    """Load timing CSV and return mean epoch time."""
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    for col in ["total_time_seconds", "total_time"]:
        if col in df.columns:
            return df[col].mean()
    return None


def load_uhlmann_result(path):
    """Load Uhlmann fidelity from result CSV."""
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    if "metric" in df.columns and "value" in df.columns:
        row = df[df["metric"] == "model_mean"]
        if len(row) > 0:
            return float(row["value"].iloc[0])
    elif "mean" in df.columns:
        return float(df["mean"].iloc[0])
    return None


def generate_latex_table(models, baseline_time):
    """Generate LaTeX table."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Model & Qubits & Tokens & Avg Epoch Time & Speedup & Test Fidelity \\",
        r"\midrule",
    ]

    for m in models:
        name = m["name"]
        qubits = m["qubits"]
        tokens = m["tokens"]

        if m["epoch_time"] is not None:
            time_str = f"{m['epoch_time']:.1f}s"
            if baseline_time and m["epoch_time"] > 0:
                speedup = baseline_time / m["epoch_time"]
                speedup_str = f"{speedup:.1f}$\\times$"
            else:
                speedup_str = "1.0$\\times$" if "element" in name.lower() else "--"
        else:
            time_str = "N/A"
            speedup_str = "--"

        if m["fidelity"] is not None:
            fid_str = f"{m['fidelity']:.3f}"
        else:
            fid_str = "N/A"

        lines.append(
            f"{name} & {qubits} & {tokens:,} & {time_str} & {speedup_str} & {fid_str} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Computational efficiency comparison: hierarchical transformers maintain constant token count (64) regardless of qubit count, enabling scalable training. The 8-qubit element-wise approach would require 65,536 tokens (infeasible).}",
            r"\label{tab:timing_comparison}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines)


def main():
    # Determine base directory
    if os.path.exists("/workspace/csvs_2"):
        base_dir = "/workspace/csvs_2"
    else:
        base_dir = "csvs_2"

    # Model configurations
    models = [
        {
            "name": "Element-wise (5q)",
            "qubits": 5,
            "tokens": 1024,
            "timing_csv": os.path.join(base_dir, "transformer_frob_timing.csv"),
            "uhlmann_csv": os.path.join(base_dir, "transformer_frob_test_uhlmann.csv"),
        },
        {
            "name": "Hierarchical (5q)",
            "qubits": 5,
            "tokens": 64,
            "timing_csv": os.path.join(
                base_dir, "hierarchical_transformer_5qubit_timing.csv"
            ),
            "uhlmann_csv": os.path.join(
                base_dir, "hierarchical_transformer_5qubit_test_uhlmann.csv"
            ),
        },
        {
            "name": "Hierarchical (8q)",
            "qubits": 8,
            "tokens": 64,
            "timing_csv": os.path.join(
                base_dir, "hierarchical_transformer_8qubit_timing.csv"
            ),
            "uhlmann_csv": os.path.join(
                base_dir, "hierarchical_transformer_8qubit_test_uhlmann.csv"
            ),
        },
    ]

    # Load data
    for m in models:
        m["epoch_time"] = load_timing_mean(m["timing_csv"])
        m["fidelity"] = load_uhlmann_result(m["uhlmann_csv"])

    # Get baseline time (element-wise 5q)
    baseline_time = models[0]["epoch_time"]

    # Generate LaTeX
    latex = generate_latex_table(models, baseline_time)

    # Print to console
    print("=" * 80)
    print("LaTeX Table")
    print("=" * 80)
    print(latex)

    # Save to file
    os.makedirs("figures", exist_ok=True)
    output_path = "figures/timing_comparison_table.tex"
    with open(output_path, "w") as f:
        f.write(latex)

    print(f"\nSaved to: {output_path}")

    # Also print a text summary
    print("\n" + "=" * 80)
    print("Text Summary")
    print("=" * 80)
    print(
        f"{'Model':<20} {'Qubits':>8} {'Tokens':>10} {'Time (s)':>12} {'Fidelity':>12}"
    )
    print("-" * 80)

    for m in models:
        time_str = f"{m['epoch_time']:.1f}" if m["epoch_time"] else "N/A"
        fid_str = f"{m['fidelity']:.4f}" if m["fidelity"] else "N/A"
        print(
            f"{m['name']:<20} {m['qubits']:>8} {m['tokens']:>10,} {time_str:>12} {fid_str:>12}"
        )


if __name__ == "__main__":
    main()
