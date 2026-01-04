"""
Test script for Cholesky-output models.
Verifies that outputs are valid density matrices.
"""

import torch
from models_3.mlp_cholesky import MLPCholeskyAutoencoder
from models_3.transformer_cholesky import TransformerCholeskyAutoencoder
from losses.frob import FrobeniusFidelityLoss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def verify_density_matrix(rho_tensor, name=""):
    """Verify that output is a valid density matrix."""
    rho = rho_tensor[:, 0] + 1j * rho_tensor[:, 1]

    # Check Hermitian
    is_hermitian = torch.allclose(rho, rho.conj().transpose(-2, -1), atol=1e-5)

    # Check trace = 1
    traces = torch.diagonal(rho, dim1=-2, dim2=-1).sum(dim=-1).real
    trace_ok = torch.allclose(traces, torch.ones_like(traces), atol=1e-5)

    # Check positive semi-definite
    eigvals = torch.linalg.eigvalsh(rho)
    min_eigval = eigvals.min().item()
    psd_ok = min_eigval >= -1e-6

    print(f"{name}:")
    print(f"  Hermitian: {is_hermitian}")
    print(f"  Trace=1: {trace_ok} (traces: {traces.tolist()})")
    print(f"  PSD: {psd_ok} (min eigenvalue: {min_eigval:.6f})")
    return is_hermitian and trace_ok and psd_ok


def main():
    print("=" * 60)
    print("Testing Cholesky-output models")
    print("=" * 60)

    # Create models
    mlp = MLPCholeskyAutoencoder(loss_fn=FrobeniusFidelityLoss())
    transformer = TransformerCholeskyAutoencoder(loss_fn=FrobeniusFidelityLoss())

    print(f"\nParameter counts:")
    print(f"  MLP Cholesky: {count_parameters(mlp):,}")
    print(f"  Transformer Cholesky: {count_parameters(transformer):,}")
    print(f"  (Original Transformer: 119,506)")

    # Test forward pass
    print(f"\nForward pass test:")
    x = torch.randn(4, 2, 32, 32)

    y_mlp = mlp(x)
    y_trans = transformer(x)

    print(f"  Input shape: {x.shape}")
    print(f"  MLP output shape: {y_mlp.shape}")
    print(f"  Transformer output shape: {y_trans.shape}")

    # Verify density matrix properties
    print(f"\nDensity matrix verification:")
    mlp_valid = verify_density_matrix(y_mlp, "MLP")
    trans_valid = verify_density_matrix(y_trans, "Transformer")

    # Test loss computation
    print(f"\nLoss computation test:")
    target = torch.randn(4, 2, 32, 32)
    loss_mlp = mlp.compute_loss(y_mlp, target)
    loss_trans = transformer.compute_loss(y_trans, target)
    print(f"  MLP loss: {loss_mlp.item():.4f}")
    print(f"  Transformer loss: {loss_trans.item():.4f}")

    # Summary
    print(f"\n" + "=" * 60)
    if mlp_valid and trans_valid:
        print("SUCCESS: All models produce valid density matrices")
    else:
        print("FAILURE: Some models produce invalid density matrices")
    print("=" * 60)


if __name__ == "__main__":
    main()