import os
import torch
import numpy as np

DATASET_DIR = "dataset_smaller"

def frob_norm(a, b):
    return torch.norm(a - b, p='fro').item()

def check_one_file(path):
    blob = torch.load(path)

    X = blob["X"]   # (N, 32, 32, 2)
    Y = blob["Y"]

    X_t = torch.tensor(X, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)

    # Compute per-sample Frobenius difference
    diffs = []
    for i in range(len(X_t)):
        # Convert 2-channel real/imag to complex matrix
        x = X_t[i, :, :, 0] + 1j * X_t[i, :, :, 1]
        y = Y_t[i, :, :, 0] + 1j * Y_t[i, :, :, 1]
        diffs.append(frob_norm(x, y))

    diffs = np.array(diffs)
    return diffs.mean(), diffs.std(), diffs.min(), diffs.max()

# ---------------------------
# MAIN
# ---------------------------
files = [f for f in os.listdir(DATASET_DIR) if f.endswith(".pt")]
files.sort()

print(f"Found {len(files)} dataset chunks.")

# Check 5 representative files: low noise, high noise, different types
sample_files = files[:5] + files[-5:]

for fname in sample_files:
    path = os.path.join(DATASET_DIR, fname)
    mean_diff, std_diff, min_diff, max_diff = check_one_file(path)

    print("\n", "-"*60)
    print(f"File: {fname}")
    print(f"Mean Frobenius difference: {mean_diff:.6f}")
    print(f"Std: {std_diff:.6f}")
    print(f"Min: {min_diff:.6f}")
    print(f"Max: {max_diff:.6f}")

