"""
Cholesky output layer for density matrix reconstruction.

Enforces physical validity by construction:
- ρ ≥ 0 (positive semi-definite) via ρ = LL†
- Tr(ρ) = 1 via normalization

For a 32×32 complex density matrix, we parameterize the lower triangular
Cholesky factor L with:
- 32 real positive diagonal entries (via softplus)
- 32×31/2 = 496 complex off-diagonal entries
- Total: 32 + 496×2 = 1024 real parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CholeskyDensityMatrix(nn.Module):
    """
    Converts raw network output to valid density matrix via Cholesky decomposition.

    Input: (B, 1024) raw values
    Output: (B, 2, 32, 32) valid density matrix [real, imag channels]
    """

    def __init__(self, matrix_size=32):
        super().__init__()
        self.n = matrix_size
        # Number of parameters for lower triangular matrix
        # Diagonal: n real values
        # Off-diagonal: n(n-1)/2 complex values = n(n-1) real values
        self.n_diag = self.n
        self.n_off_diag = self.n * (self.n - 1) // 2
        self.total_params = self.n_diag + 2 * self.n_off_diag  # 1024 for n=32

    def forward(self, x):
        """
        Args:
            x: (B, 1024) raw network output
        Returns:
            rho: (B, 2, 32, 32) valid density matrix
        """
        B = x.shape[0]

        # Split into diagonal and off-diagonal parts
        diag_raw = x[:, :self.n_diag]  # (B, 32)
        off_diag_raw = x[:, self.n_diag:]  # (B, 992)

        # Diagonal: real and positive via softplus with minimum floor
        # Floor prevents trace from becoming too small, which would cause
        # gradient explosion during backprop through the trace normalization
        diag = F.softplus(diag_raw) + 1e-4  # (B, 32)

        # Off-diagonal: complex values
        off_diag_real = off_diag_raw[:, :self.n_off_diag]  # (B, 496)
        off_diag_imag = off_diag_raw[:, self.n_off_diag:]  # (B, 496)
        off_diag = torch.complex(off_diag_real, off_diag_imag)  # (B, 496)

        # Build lower triangular matrix L
        L = torch.zeros(B, self.n, self.n, dtype=torch.complex64, device=x.device)

        # Fill diagonal
        L[:, range(self.n), range(self.n)] = diag.to(torch.complex64)

        # Fill lower triangular (below diagonal)
        tril_indices = torch.tril_indices(self.n, self.n, offset=-1)
        L[:, tril_indices[0], tril_indices[1]] = off_diag

        # Compute ρ = LL†
        rho = torch.bmm(L, L.conj().transpose(-2, -1))

        # Normalize trace to 1
        trace = torch.diagonal(rho, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)
        rho = rho / (trace.real + 1e-8)

        # Convert to (B, 2, 32, 32) format
        rho_out = torch.stack([rho.real, rho.imag], dim=1)

        return rho_out.float()


def count_cholesky_params(matrix_size=32):
    """Calculate number of parameters needed for Cholesky factorization."""
    n = matrix_size
    n_diag = n
    n_off_diag = n * (n - 1) // 2
    return n_diag + 2 * n_off_diag


if __name__ == "__main__":
    # Test the module
    layer = CholeskyDensityMatrix(32)
    print(f"Required input size: {layer.total_params}")

    # Test forward pass
    x = torch.randn(4, 1024)
    rho = layer(x)
    print(f"Input: {x.shape}, Output: {rho.shape}")

    # Verify properties
    rho_complex = rho[:, 0] + 1j * rho[:, 1]

    # Check Hermitian
    is_hermitian = torch.allclose(rho_complex, rho_complex.conj().transpose(-2, -1), atol=1e-5)
    print(f"Is Hermitian: {is_hermitian}")

    # Check trace = 1
    traces = torch.diagonal(rho_complex, dim1=-2, dim2=-1).sum(dim=-1)
    print(f"Traces: {traces.real}")

    # Check positive semi-definite (all eigenvalues >= 0)
    eigvals = torch.linalg.eigvalsh(rho_complex)
    min_eigval = eigvals.min().item()
    print(f"Min eigenvalue: {min_eigval:.6f} (should be >= 0)")