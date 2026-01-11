"""
Demonstrate that row-wise output projection prevents global mixing.

The Cholesky factorization requires ALL 1024 parameters to be coordinated,
but the current architecture computes each row independently.
"""

import torch
import torch.nn as nn

def main():
    print("="*70)
    print("TESTING OUTPUT MIXING PROBLEM")
    print("="*70)

    batch_size = 4
    num_rows = 32
    embed_dim = 64

    print(f"\nSetup:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num rows: {num_rows}")
    print(f"  Embed dim: {embed_dim}")

    # Simulate decoder output (after dropping CLS token)
    row_output = torch.randn(batch_size, num_rows, embed_dim)
    print(f"\nDecoder output (after dropping CLS): {row_output.shape}")

    # Current architecture: per-token projection
    print("\n" + "-"*70)
    print("CURRENT ARCHITECTURE: Per-Token Projection")
    print("-"*70)

    output_proj = nn.Linear(embed_dim, 32)
    chol_params = output_proj(row_output)
    print(f"\nAfter Linear(embed_dim={embed_dim}, out=32): {chol_params.shape}")
    chol_params_flat = chol_params.reshape(batch_size, -1)
    print(f"Flattened for Cholesky: {chol_params_flat.shape}")

    print("\nProblem with per-token projection:")
    print("  • Each row's 32 output values depend ONLY on that row's embedding")
    print("  • Row 0 output = f(row_0_embedding)")
    print("  • Row 1 output = f(row_1_embedding)")
    print("  • ...")
    print("  • No final mixing across rows!")

    print("\nWhy this breaks Cholesky factorization:")
    print("  • Cholesky needs ρ = LL† where L is lower triangular")
    print("  • Element ρ[i,j] depends on dot product of row i and row j of L")
    print("  • If L[i,:] and L[j,:] are computed independently, they can't coordinate")
    print("  • Result: Degenerate solutions like uniform diagonal (I/32)")

    # What should happen
    print("\n" + "-"*70)
    print("WHAT SHOULD HAPPEN: Global Mixing")
    print("-"*70)

    print("\nOption 1: Use CLS token for global prediction")
    # Simulate using CLS token
    cls_output = torch.randn(batch_size, 1, embed_dim)
    global_proj = nn.Linear(embed_dim, 1024)
    chol_params_global = global_proj(cls_output.squeeze(1))
    print(f"\n  CLS output: {cls_output.shape}")
    print(f"  After Linear({embed_dim}, 1024): {chol_params_global.shape}")
    print(f"  ✓ All 1024 params computed from same global representation")

    print("\nOption 2: Flatten all row embeddings then project")
    row_flat = row_output.reshape(batch_size, -1)
    flatten_proj = nn.Linear(num_rows * embed_dim, 1024)
    chol_params_flat_proj = flatten_proj(row_flat)
    print(f"\n  Flattened rows: {row_flat.shape}")
    print(f"  After Linear({num_rows * embed_dim}, 1024): {chol_params_flat_proj.shape}")
    print(f"  ✓ All 1024 params depend on all row embeddings")

    print("\nOption 3: Use attention pooling across rows")
    pooling = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
    query = torch.randn(batch_size, 1, embed_dim)
    pooled, _ = pooling(query, row_output, row_output)
    pooled_proj = nn.Linear(embed_dim, 1024)
    chol_params_pooled = pooled_proj(pooled.squeeze(1))
    print(f"\n  Pooled via attention: {pooled.shape}")
    print(f"  After Linear({embed_dim}, 1024): {chol_params_pooled.shape}")
    print(f"  ✓ All 1024 params from globally-attended representation")

    print("\n" + "-"*70)
    print("CURRENT SITUATION")
    print("-"*70)

    print("\nThe model:")
    print("  1. Has CLS token that aggregates global info")
    print("  2. Immediately throws it away (line 105: skip CLS)")
    print("  3. Projects each row independently")
    print("  4. Cannot coordinate Cholesky parameters")
    print("  5. Learns to output uniform/maximally mixed state")

    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print("\nUse CLS token for global prediction:")
    print("  cls_output = dec[:, 0, :]  # (B, embed_dim)")
    print("  chol_params = global_proj(cls_output)  # (B, 1024)")
    print("\nThis way all Cholesky parameters are computed jointly.")

if __name__ == "__main__":
    main()
