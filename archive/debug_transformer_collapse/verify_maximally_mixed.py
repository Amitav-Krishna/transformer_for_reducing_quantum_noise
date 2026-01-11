"""
Verify that transformer is outputting the maximally mixed state I/32.

Expected behavior:
- Maximally mixed state has diagonal = 1/32 everywhere
- Fidelity with any state should be approximately 1/32 = 0.03125
- Observed fidelity: ~0.031

This script loads the checkpoint and checks actual outputs.
"""

import torch
import sys
sys.path.insert(0, '/home/work/codage/transformer_qnr')

from models_3.transformer_cholesky import TransformerCholeskyAutoencoder
from losses.frob import FrobeniusFidelityLoss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = TransformerCholeskyAutoencoder(loss_fn=FrobeniusFidelityLoss()).to(device)
    ckpt_path = "/home/work/codage/transformer_qnr/checkpoints_3/transformer_cholesky/best.pt"

    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded checkpoint: {ckpt_path}")
    except FileNotFoundError:
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        return

    model.eval()

    # Generate diverse random inputs
    print("\nTesting with 100 random inputs...")
    n_samples = 100
    all_traces = []
    all_diagonals = []

    with torch.no_grad():
        for i in range(10):  # 10 batches of 10
            x = torch.randn(10, 2, 32, 32).to(device)
            output = model(x)

            # Convert to complex
            rho = output[:, 0] + 1j * output[:, 1]

            # Get diagonal elements
            for j in range(10):
                diag = torch.diagonal(rho[j]).real.cpu().numpy()
                all_diagonals.append(diag)
                trace = diag.sum()
                all_traces.append(trace)

    all_diagonals = torch.tensor(all_diagonals)
    all_traces = torch.tensor(all_traces)

    print(f"\nStatistics over {n_samples} random inputs:")
    print(f"  Trace mean: {all_traces.mean():.6f} (should be 1.0)")
    print(f"  Trace std:  {all_traces.std():.6f}")
    print()

    # Check if diagonal is uniform (maximally mixed)
    diag_mean = all_diagonals.mean(dim=0)
    diag_std = all_diagonals.std(dim=0)

    print(f"Diagonal element statistics:")
    print(f"  Mean across all positions: {diag_mean.mean():.6f} (expected for I/32: {1/32:.6f})")
    print(f"  Std across all positions:  {diag_std.mean():.6f}")
    print()

    # Check if all diagonal elements are similar (uniformity test)
    overall_mean = all_diagonals.mean()
    overall_std = all_diagonals.std()

    print(f"Overall diagonal uniformity:")
    print(f"  Global mean: {overall_mean:.6f}")
    print(f"  Global std:  {overall_std:.6f}")
    print()

    # Compare to maximally mixed state
    expected_diag = 1.0 / 32.0
    deviation_from_maximally_mixed = abs(overall_mean - expected_diag)

    print(f"Comparison to maximally mixed state (I/32):")
    print(f"  Expected diagonal value: {expected_diag:.6f}")
    print(f"  Actual mean diagonal:    {overall_mean:.6f}")
    print(f"  Deviation:               {deviation_from_maximally_mixed:.6f}")
    print()

    if deviation_from_maximally_mixed < 0.001 and overall_std < 0.001:
        print("✓ CONFIRMED: Model is outputting the maximally mixed state!")
        print("  All diagonal elements are uniformly ~1/32")
        print("  This explains the constant ~0.031 fidelity")
    else:
        print("✗ Model output is not exactly maximally mixed")
        print("  But may still be very close to uniform")

    # Sample a few actual diagonal vectors
    print(f"\nSample diagonal vectors (first 5 samples):")
    for i in range(min(5, len(all_diagonals))):
        diag = all_diagonals[i]
        print(f"  Sample {i+1}: min={diag.min():.4f}, max={diag.max():.4f}, "
              f"mean={diag.mean():.4f}, std={diag.std():.4f}")

if __name__ == "__main__":
    main()
