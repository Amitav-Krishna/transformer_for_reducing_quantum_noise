"""
Compare 5-qubit vs 8-qubit hierarchical transformers to demonstrate scaling.

Key insight: Token count stays constant (64 tokens) while matrix size grows 64x.
This proves the hierarchical approach scales with qubit count.

Output: scaling_analysis.csv
"""

import os
import csv
import pandas as pd
from pathlib import Path


def load_timing_csv(path):
    """Load timing CSV and return statistics."""
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    time_col = None
    for col in ["total_time_seconds", "total_time"]:
        if col in df.columns:
            time_col = col
            break

    if time_col is None:
        return None

    return {
        "mean": df[time_col].mean(),
        "std": df[time_col].std(),
        "min": df[time_col].min(),
        "max": df[time_col].max(),
    }


def main():
    # Define paths
    if os.path.exists("/workspace/csvs_2"):
        base_dir = "/workspace/csvs_2"
    else:
        base_dir = "csvs_2"

    # Model configurations
    models = {
        "hierarchical_5qubit": {
            "timing_csv": os.path.join(
                base_dir, "hierarchical_transformer_5qubit_timing.csv"
            ),
            "qubits": 5,
            "matrix_size": 32,
            "patch_size": 4,
            "tokens": 64,  # (32/4)^2
            "batch_size": 8,
            "params": 534592,
        },
        "hierarchical_8qubit": {
            "timing_csv": os.path.join(
                base_dir, "hierarchical_transformer_8qubit_timing.csv"
            ),
            "qubits": 8,
            "matrix_size": 256,
            "patch_size": 32,
            "tokens": 64,  # (256/32)^2
            "batch_size": 4,
            "params": 1611072,
        },
    }

    results = []

    for model_name, config in models.items():
        timing = load_timing_csv(config["timing_csv"])

        # Calculate memory per sample (bytes, float32)
        matrix_elements = config["matrix_size"] ** 2 * 2  # 2 channels (real/imag)
        memory_per_sample_kb = matrix_elements * 4 / 1024  # float32 = 4 bytes

        # Attention complexity
        attention_ops = config["tokens"] ** 2  # O(n^2) for self-attention

        # What element-wise would require
        elementwise_tokens = config["matrix_size"] ** 2
        elementwise_attention_ops = elementwise_tokens**2

        result = {
            "model_name": model_name,
            "qubits": config["qubits"],
            "matrix_size": f"{config['matrix_size']}x{config['matrix_size']}",
            "patch_size": f"{config['patch_size']}x{config['patch_size']}",
            "tokens": config["tokens"],
            "batch_size": config["batch_size"],
            "params": config["params"],
            "memory_per_sample_kb": memory_per_sample_kb,
            "attention_ops": attention_ops,
            "elementwise_tokens": elementwise_tokens,
            "elementwise_attention_ops": elementwise_attention_ops,
            "compression_ratio": elementwise_tokens / config["tokens"],
        }

        if timing:
            result["avg_epoch_time"] = timing["mean"]
            result["epoch_time_std"] = timing["std"]
        else:
            result["avg_epoch_time"] = None
            result["epoch_time_std"] = None

        results.append(result)

    # Print results
    print("=" * 100)
    print("SCALING ANALYSIS: 5-qubit vs 8-qubit Hierarchical Transformers")
    print("=" * 100)
    print()

    print(f"{'Metric':<30} {'5-qubit':>20} {'8-qubit':>20} {'Ratio':>15}")
    print("-" * 100)

    r5 = results[0]
    r8 = results[1]

    metrics = [
        ("Qubits", "qubits", ""),
        ("Matrix Size", "matrix_size", ""),
        ("Patch Size", "patch_size", ""),
        ("Tokens (hierarchical)", "tokens", ""),
        ("Tokens (element-wise)", "elementwise_tokens", ""),
        ("Compression Ratio", "compression_ratio", "x"),
        ("Attention Ops (hierarchical)", "attention_ops", ""),
        ("Attention Ops (element-wise)", "elementwise_attention_ops", ""),
        ("Batch Size", "batch_size", ""),
        ("Memory/Sample (KB)", "memory_per_sample_kb", ""),
        ("Parameters", "params", ""),
        ("Avg Epoch Time (s)", "avg_epoch_time", ""),
    ]

    for label, key, suffix in metrics:
        v5 = r5.get(key)
        v8 = r8.get(key)

        if v5 is None or v8 is None:
            ratio_str = "N/A"
        elif isinstance(v5, str):
            ratio_str = "-"
        else:
            ratio = v8 / v5 if v5 != 0 else float("inf")
            ratio_str = f"{ratio:.2f}x"

        v5_str = (
            f"{v5:,.0f}{suffix}"
            if isinstance(v5, (int, float)) and v5 is not None
            else str(v5)
        )
        v8_str = (
            f"{v8:,.0f}{suffix}"
            if isinstance(v8, (int, float)) and v8 is not None
            else str(v8)
        )

        print(f"{label:<30} {v5_str:>20} {v8_str:>20} {ratio_str:>15}")

    # Key insights
    print()
    print("=" * 100)
    print("KEY INSIGHTS FOR PAPER")
    print("=" * 100)
    print()
    print("1. TOKEN COUNT IS CONSTANT:")
    print(f"   - 5-qubit: {r5['tokens']} tokens")
    print(f"   - 8-qubit: {r8['tokens']} tokens")
    print(f"   -> Attention complexity O(64^2) = O(4096) regardless of qubit count!")
    print()
    print("2. ELEMENT-WISE WOULD BE INFEASIBLE:")
    print(
        f"   - 5-qubit element-wise: {r5['elementwise_tokens']:,} tokens, {r5['elementwise_attention_ops']:,} attention ops"
    )
    print(
        f"   - 8-qubit element-wise: {r8['elementwise_tokens']:,} tokens, {r8['elementwise_attention_ops']:,.0f} attention ops"
    )
    print(
        f"   -> 8-qubit element-wise requires {r8['elementwise_attention_ops'] / r8['attention_ops']:,.0f}x more ops than hierarchical!"
    )
    print()
    print("3. MEMORY SCALING:")
    print(f"   - 5-qubit: {r5['memory_per_sample_kb']:.1f} KB/sample")
    print(f"   - 8-qubit: {r8['memory_per_sample_kb']:.1f} KB/sample")
    print(
        f"   -> Only {r8['memory_per_sample_kb'] / r5['memory_per_sample_kb']:.0f}x memory increase (linear in matrix size)"
    )
    print()

    if r5["avg_epoch_time"] and r8["avg_epoch_time"]:
        time_ratio = r8["avg_epoch_time"] / r5["avg_epoch_time"]
        print("4. TRAINING TIME SCALING:")
        print(f"   - 5-qubit: {r5['avg_epoch_time']:.1f}s/epoch")
        print(f"   - 8-qubit: {r8['avg_epoch_time']:.1f}s/epoch")
        print(f"   -> Only {time_ratio:.1f}x slower despite 64x larger matrices!")
        print()

    # Save to CSV
    output_path = "scaling_analysis.csv"
    with open(output_path, "w", newline="") as f:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
