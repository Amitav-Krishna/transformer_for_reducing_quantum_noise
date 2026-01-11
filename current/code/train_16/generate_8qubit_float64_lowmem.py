#!/usr/bin/env python3
"""
Low-memory version of 8-qubit float64 dataset generation.
Uses only 4 workers to fit in 32GB RAM.
"""
import sys
sys.path.insert(0, "/workspace")

from train_16.generate_8qubit_float64_parallel import generate_8qubit_dataset_parallel

if __name__ == "__main__":
    generate_8qubit_dataset_parallel(
        output_dir="dataset_8qubit_float64",
        num_samples_per_cell=5000,
        num_workers=4,  # Low memory mode
        seed=42,
    )
