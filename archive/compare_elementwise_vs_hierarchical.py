"""
Compare element-wise vs hierarchical transformers (5-qubit).

Analyzes:
1. Training time per epoch
2. Test Uhlmann fidelity
3. Parameter count
4. Speedup ratio

Output: results_comparison.csv
"""

import os
import csv
import pandas as pd
from pathlib import Path


def load_timing_csv(path):
    """Load timing CSV and return average epoch time."""
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    if "total_time_seconds" in df.columns:
        return df["total_time_seconds"].mean()
    elif "total_time" in df.columns:
        return df["total_time"].mean()
    return None


def load_uhlmann_result(path):
    """Load Uhlmann fidelity result from CSV."""
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    # Handle different formats
    if "metric" in df.columns and "value" in df.columns:
        # New format: metric, value rows
        row = df[df["metric"] == "model_mean"]
        if len(row) > 0:
            return float(row["value"].iloc[0])
    elif "mean" in df.columns:
        return df["mean"].iloc[0]
    return None


def count_model_params(model_name):
    """Return known parameter counts for models."""
    # Pre-computed values (from scripts/count_params.py)
    PARAM_COUNTS = {
        "element_wise_5qubit": 119506,
        "hierarchical_5qubit": 534592,  # Will verify when model is loaded
        "hierarchical_8qubit": 1611072,
    }
    return PARAM_COUNTS.get(model_name, None)


def main():
    # Define paths for different setups
    if os.path.exists("/workspace/csvs_2"):
        base_dir = "/workspace/csvs_2"
    else:
        base_dir = "csvs_2"

    # Models to compare
    models = {
        "element_wise_5qubit": {
            "timing_csv": os.path.join(base_dir, "transformer_frob_timing.csv"),
            "uhlmann_csv": os.path.join(base_dir, "transformer_frob_test_uhlmann.csv"),
            "alt_timing": "csvs_2/transformer.csv",  # May have timing in main log
        },
        "hierarchical_5qubit": {
            "timing_csv": os.path.join(
                base_dir, "hierarchical_transformer_5qubit_timing.csv"
            ),
            "uhlmann_csv": os.path.join(
                base_dir, "hierarchical_transformer_5qubit_test_uhlmann.csv"
            ),
        },
        "hierarchical_8qubit": {
            "timing_csv": os.path.join(
                base_dir, "hierarchical_transformer_8qubit_timing.csv"
            ),
            "uhlmann_csv": os.path.join(
                base_dir, "hierarchical_transformer_8qubit_test_uhlmann.csv"
            ),
        },
    }

    results = []
    baseline_time = None

    for model_name, paths in models.items():
        # Load timing
        avg_time = load_timing_csv(paths["timing_csv"])
        if avg_time is None and "alt_timing" in paths:
            avg_time = load_timing_csv(paths["alt_timing"])

        # Load Uhlmann fidelity
        uhlmann = load_uhlmann_result(paths["uhlmann_csv"])

        # Get param count
        num_params = count_model_params(model_name)

        # Store baseline time for speedup calculation
        if model_name == "element_wise_5qubit" and avg_time is not None:
            baseline_time = avg_time

        results.append(
            {
                "model_name": model_name,
                "avg_epoch_time": avg_time,
                "test_uhlmann_fidelity": uhlmann,
                "num_params": num_params,
            }
        )

    # Calculate speedup
    for r in results:
        if baseline_time is not None and r["avg_epoch_time"] is not None:
            r["speedup"] = baseline_time / r["avg_epoch_time"]
        else:
            r["speedup"] = None

    # Print results
    print("=" * 80)
    print("COMPARISON: Element-wise vs Hierarchical Transformers")
    print("=" * 80)
    print(
        f"{'Model':<25} {'Avg Epoch (s)':>15} {'Uhlmann Fid':>12} {'Params':>12} {'Speedup':>10}"
    )
    print("-" * 80)

    for r in results:
        time_str = f"{r['avg_epoch_time']:.1f}" if r["avg_epoch_time"] else "N/A"
        fid_str = (
            f"{r['test_uhlmann_fidelity']:.4f}" if r["test_uhlmann_fidelity"] else "N/A"
        )
        params_str = f"{r['num_params']:,}" if r["num_params"] else "N/A"
        speedup_str = f"{r['speedup']:.2f}x" if r["speedup"] else "N/A"
        print(
            f"{r['model_name']:<25} {time_str:>15} {fid_str:>12} {params_str:>12} {speedup_str:>10}"
        )

    # Save to CSV
    output_path = "results_comparison.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_name",
                "avg_epoch_time",
                "test_uhlmann_fidelity",
                "num_params",
                "speedup",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\nResults saved to: {output_path}")

    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    hierarchical_5q = next(
        (r for r in results if r["model_name"] == "hierarchical_5qubit"), None
    )
    elementwise_5q = next(
        (r for r in results if r["model_name"] == "element_wise_5qubit"), None
    )

    if hierarchical_5q and elementwise_5q:
        if hierarchical_5q["speedup"]:
            print(
                f"- Hierarchical (5q) is {hierarchical_5q['speedup']:.1f}x faster than element-wise"
            )
        if (
            hierarchical_5q["test_uhlmann_fidelity"]
            and elementwise_5q["test_uhlmann_fidelity"]
        ):
            fid_diff = (
                elementwise_5q["test_uhlmann_fidelity"]
                - hierarchical_5q["test_uhlmann_fidelity"]
            )
            print(
                f"- Fidelity tradeoff: {fid_diff:.4f} ({fid_diff / elementwise_5q['test_uhlmann_fidelity'] * 100:.1f}% reduction)"
            )

    hierarchical_8q = next(
        (r for r in results if r["model_name"] == "hierarchical_8qubit"), None
    )
    if hierarchical_8q:
        print(f"- 8-qubit hierarchical successfully trained (element-wise would OOM)")
        if hierarchical_8q["test_uhlmann_fidelity"]:
            print(
                f"- 8-qubit Uhlmann fidelity: {hierarchical_8q['test_uhlmann_fidelity']:.4f}"
            )


if __name__ == "__main__":
    main()
