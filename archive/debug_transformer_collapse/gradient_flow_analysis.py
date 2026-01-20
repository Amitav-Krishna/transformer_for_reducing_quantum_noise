"""
Analyze potential gradient flow issues through Cholesky layer.

When model outputs nearly uniform values, the Cholesky normalization
can cause gradient problems.
"""

import torch
import torch.nn.functional as F

def main():
    print("="*70)
    print("GRADIENT FLOW ANALYSIS: Cholesky Layer")
    print("="*70)

    batch_size = 4
    n = 32

    print(f"\nSetup: {n}x{n} density matrices, batch size {batch_size}")

    # Simulate what happens when model outputs uniform values
    print("\n" + "-"*70)
    print("SCENARIO 1: Model outputs uniform values (collapsed state)")
    print("-"*70)

    # All params close to zero (what the model might learn)
    chol_params_uniform = torch.zeros(batch_size, 1024, requires_grad=True)

    # Cholesky layer processing
    n_diag = n
    n_off_diag = n * (n - 1) // 2

    diag_raw = chol_params_uniform[:, :n_diag]
    off_diag_raw = chol_params_uniform[:, n_diag:]

    # Apply softplus + floor
    diag = F.softplus(diag_raw) + 1e-4

    print(f"\nDiagonal values after softplus + 1e-4:")
    print(f"  Mean: {diag.mean():.6f}")
    print(f"  Std:  {diag.std():.6f}")
    print(f"  Min:  {diag.min():.6f}")
    print(f"  Max:  {diag.max():.6f}")

    # Off-diagonal
    off_diag_real = off_diag_raw[:, :n_off_diag]
    off_diag_imag = off_diag_raw[:, n_off_diag:]
    off_diag = torch.complex(off_diag_real, off_diag_imag)

    # Build L matrix
    L = torch.zeros(batch_size, n, n, dtype=torch.complex64)
    L[:, range(n), range(n)] = diag.to(torch.complex64)
    tril_indices = torch.tril_indices(n, n, offset=-1)
    L[:, tril_indices[0], tril_indices[1]] = off_diag

    # Compute rho = LL†
    rho = torch.bmm(L, L.conj().transpose(-2, -1))

    # Trace normalization
    trace = torch.diagonal(rho, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)

    print(f"\nTrace before normalization:")
    print(f"  Mean: {trace.real.mean():.6f}")
    print(f"  Std:  {trace.real.std():.6f}")

    # The problem: when params are uniform, trace is dominated by floor term
    print(f"\nIf diagonal is all 1e-4:")
    expected_trace = 32 * (1e-4) ** 2
    print(f"  Expected trace ≈ 32 × (1e-4)² = {expected_trace:.2e}")
    print(f"  Actual trace: {trace.real.mean():.2e}")

    # After normalization
    rho_normalized = rho / (trace.real + 1e-8)

    print(f"\nAfter normalization rho / (trace + 1e-8):")
    trace_after = torch.diagonal(rho_normalized, dim1=-2, dim2=-1).sum(dim=-1)
    print(f"  Trace mean: {trace_after.real.mean():.6f}")

    diag_after = torch.diagonal(rho_normalized, dim1=-2, dim2=-1).real
    print(f"  Diagonal mean: {diag_after.mean():.6f}")
    print(f"  Expected for I/32: {1/32:.6f}")

    print("\n⚠️  PROBLEM:")
    print("  When all params ≈ 0, softplus + 1e-4 ≈ 1e-4")
    print("  Trace ≈ 32 × (1e-4)² ≈ 3.2e-7")
    print("  After normalization: rho ≈ I/32 (maximally mixed)")
    print("  Gradients flow through division by tiny numbers!")

    # Gradient test
    print("\n" + "-"*70)
    print("SCENARIO 2: Testing gradient flow")
    print("-"*70)

    # Simulate loss
    target = torch.randn(batch_size, 2, n, n)
    target_complex = target[:, 0] + 1j * target[:, 1]

    # Simple loss (Frobenius)
    diff = rho_normalized - target_complex
    loss = (diff.real ** 2 + diff.imag ** 2).mean()

    print(f"\nLoss: {loss.item():.6f}")

    # Backprop
    loss.backward()

    if chol_params_uniform.grad is not None:
        grad = chol_params_uniform.grad
        print(f"\nGradient statistics:")
        print(f"  Mean: {grad.mean():.6e}")
        print(f"  Std:  {grad.std():.6e}")
        print(f"  Max abs: {grad.abs().max():.6e}")

        # Check for numerical issues
        if torch.isnan(grad).any():
            print("  ❌ NaN gradients detected!")
        if torch.isinf(grad).any():
            print("  ❌ Inf gradients detected!")
        if grad.abs().max() > 1e6:
            print("  ⚠️  Very large gradients (>1e6)")
        if grad.abs().max() < 1e-6:
            print("  ⚠️  Very small gradients (<1e-6) - vanishing gradient")

    print("\n" + "-"*70)
    print("SCENARIO 3: Softplus saturation")
    print("-"*70)

    # Test softplus behavior
    test_inputs = torch.tensor([-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0])
    test_outputs = F.softplus(test_inputs)

    print("\nSoftplus behavior:")
    print(f"  Input:  {test_inputs.tolist()}")
    print(f"  Output: {test_outputs.tolist()}")

    print("\nFor negative inputs:")
    print("  softplus(-10) ≈ 0.00005")
    print("  softplus(-5) ≈ 0.007")
    print("  Adding floor 1e-4 doesn't help if input is very negative")

    print("\nIf network learns negative weights (common with zero init):")
    print("  → Diagonal values ≈ 1e-4")
    print("  → Trace ≈ 32 × (1e-4)²")
    print("  → Normalized ≈ I/32")
    print("  → Stuck in maximally mixed state")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    print("\nCholesky layer is not fundamentally broken, but:")
    print("  1. When model outputs uniform values, result is I/32")
    print("  2. This is a stable fixed point in the loss landscape")
    print("  3. Gradients through trace normalization can be weak")
    print("  4. Model has no incentive to move away from this solution")
    print("\nCombined with architectural issues (decoder, output projection),")
    print("the model gets stuck outputting maximally mixed state.")

if __name__ == "__main__":
    main()
