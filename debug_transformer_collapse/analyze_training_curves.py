"""
Analyze training curves to show that transformer never learned.

Compares transformer vs MLP training progression.
"""

import pandas as pd
import sys

def main():
    # Read CSV files
    transformer_csv = "/home/work/codage/transformer_qnr/csvs_3/transformer_cholesky.csv"
    mlp_csv = "/home/work/codage/transformer_qnr/csvs_3/mlp_cholesky.csv"

    print("="*70)
    print("TRAINING CURVE ANALYSIS: Transformer vs MLP Cholesky Models")
    print("="*70)

    # Parse validation losses
    transformer_val = []
    mlp_val = []

    print("\nParsing transformer validation losses...")
    with open(transformer_csv, 'r') as f:
        for line in f:
            if ',val,' in line:
                parts = line.strip().split(',')
                epoch = int(parts[0])
                val_loss = float(parts[-1])
                transformer_val.append((epoch, val_loss))

    print(f"Found {len(transformer_val)} validation epochs")

    print("\nParsing MLP validation losses...")
    with open(mlp_csv, 'r') as f:
        for line in f:
            if ',val,' in line:
                parts = line.strip().split(',')
                epoch = int(parts[0])
                val_loss = float(parts[-1])
                mlp_val.append((epoch, val_loss))

    print(f"Found {len(mlp_val)} validation epochs")

    # Transformer analysis
    print("\n" + "="*70)
    print("TRANSFORMER CHOLESKY")
    print("="*70)

    if len(transformer_val) > 0:
        first_epoch, first_loss = transformer_val[0]
        last_epoch, last_loss = transformer_val[-1]
        best_loss = min([loss for _, loss in transformer_val])
        best_epoch = [epoch for epoch, loss in transformer_val if loss == best_loss][0]

        print(f"\nEpoch {first_epoch:3d}: Val loss = {first_loss:.6f}")
        print(f"Epoch {last_epoch:3d}: Val loss = {last_loss:.6f}")
        print(f"\nBest epoch: {best_epoch} with loss {best_loss:.6f}")
        print(f"Total improvement: {first_loss - last_loss:.6f}")
        print(f"Improvement to best: {first_loss - best_loss:.6f}")

        # Check if essentially flat
        improvement = first_loss - best_loss
        if improvement < 0.001:
            print("\n⚠️  CRITICAL: Training is FLAT - no meaningful improvement!")
            print(f"   Improvement < 0.001 indicates model never learned")

        # Show epoch-by-epoch (first 10)
        print(f"\nFirst 10 epochs:")
        for i, (epoch, loss) in enumerate(transformer_val[:10]):
            improvement = transformer_val[0][1] - loss
            print(f"  Epoch {epoch:2d}: {loss:.6f} (Δ = {improvement:+.6f})")

    # MLP analysis
    print("\n" + "="*70)
    print("MLP CHOLESKY")
    print("="*70)

    if len(mlp_val) > 0:
        first_epoch, first_loss = mlp_val[0]
        last_epoch, last_loss = mlp_val[-1]
        best_loss = min([loss for _, loss in mlp_val])
        best_epoch = [epoch for epoch, loss in mlp_val if loss == best_loss][0]

        print(f"\nEpoch {first_epoch:3d}: Val loss = {first_loss:.6f}")
        print(f"Epoch {last_epoch:3d}: Val loss = {last_loss:.6f}")
        print(f"\nBest epoch: {best_epoch} with loss {best_loss:.6f}")
        print(f"Total improvement: {first_loss - last_loss:.6f}")
        print(f"Improvement to best: {first_loss - best_loss:.6f}")

        # Show epoch-by-epoch (first 10)
        print(f"\nFirst 10 epochs:")
        for i, (epoch, loss) in enumerate(mlp_val[:10]):
            improvement = mlp_val[0][1] - loss
            print(f"  Epoch {epoch:2d}: {loss:.6f} (Δ = {improvement:+.6f})")

    # Comparison
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)

    if len(transformer_val) > 0 and len(mlp_val) > 0:
        trans_improvement = transformer_val[0][1] - min([l for _, l in transformer_val])
        mlp_improvement = mlp_val[0][1] - min([l for _, l in mlp_val])

        print(f"\nTransformer total improvement: {trans_improvement:.6f}")
        print(f"MLP total improvement:         {mlp_improvement:.6f}")
        print(f"\nMLP learned {mlp_improvement / max(trans_improvement, 1e-10):.1f}x more than transformer")

        print("\nCONCLUSION:")
        if trans_improvement < 0.001:
            print("  ❌ Transformer: No learning occurred (flat loss)")
        else:
            print("  ✓ Transformer: Some learning occurred")

        if mlp_improvement > 0.01:
            print("  ✓ MLP: Clear learning occurred (continuous improvement)")
        else:
            print("  ❌ MLP: Minimal learning")

if __name__ == "__main__":
    main()
