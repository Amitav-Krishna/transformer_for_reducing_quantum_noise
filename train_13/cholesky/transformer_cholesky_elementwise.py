"""
Element-wise Transformer with Cholesky output for density matrix denoising.

This addresses the reviewer critique: the original Cholesky Transformer used
global pooling, which is architecturally different from the post-hoc version.

This version maintains ELEMENT-WISE output structure:
- Post-hoc: 1024 tokens → each outputs 2 values → 2048 total → post-hoc projection
- Element-wise Cholesky: 1024 tokens → each outputs 1 value → 1024 total → Cholesky layer

This is a fair comparison since both use the same attention mechanism and
element-wise output structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CholeskyDensityMatrix(nn.Module):
    """
    Converts 1024 raw values to valid density matrix via Cholesky decomposition.

    For a 32×32 complex density matrix, the Cholesky factor L has:
    - 32 real positive diagonal entries (via softplus)
    - 32×31/2 = 496 complex off-diagonal entries (992 real values)
    - Total: 32 + 992 = 1024 real parameters
    """

    def __init__(self, matrix_size=32):
        super().__init__()
        self.n = matrix_size
        self.n_diag = self.n
        self.n_off_diag = self.n * (self.n - 1) // 2

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

        # Diagonal: real and positive via softplus
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

        # Compute rho = LL†
        rho = torch.bmm(L, L.conj().transpose(-2, -1))

        # Normalize trace to 1
        trace = torch.diagonal(rho, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)
        rho = rho / (trace.real + 1e-8)

        # Convert to (B, 2, 32, 32) format
        rho_out = torch.stack([rho.real, rho.imag], dim=1)

        return rho_out.float()


class TransformerCholeskyElementwise(nn.Module):
    """
    Transformer autoencoder with ELEMENT-WISE Cholesky output.

    Same architecture as post-hoc Transformer (train_v8/transformer.py):
    - 1024 tokens, embed_dim=32, ffn_dim=64, 4 heads, 4 layers
    - Bottleneck: 32 -> 16 -> 32

    Key difference: each token outputs 1 value (not 2), giving 1024 Cholesky
    parameters that are converted to a valid density matrix.

    This maintains the element-wise attention structure while using hard
    physical constraints.

    Parameters: ~119k (same as post-hoc version, just output_proj is 32->1 not 32->2)
    """

    def __init__(self, loss_fn=None):
        super().__init__()
        self.loss_fn = loss_fn

        self.seq_len = 1024
        self.input_dim = 2
        self.embed_dim = 32
        self.ffn_dim = 64
        self.num_heads = 4
        self.layers = 4

        self.input_proj = nn.Linear(2, self.embed_dim)
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.seq_len, self.embed_dim)
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ffn_dim,
            batch_first=True,
            activation="gelu",
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.layers)

        self.down = nn.Linear(self.embed_dim, 16)
        self.up = nn.Linear(16, self.embed_dim)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ffn_dim,
            batch_first=True,
            activation="gelu",
            dropout=0.1,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=self.layers)

        # Element-wise output: each token outputs 1 value for Cholesky params
        self.output_proj = nn.Linear(self.embed_dim, 1)

        # Cholesky output layer
        self.cholesky = CholeskyDensityMatrix(matrix_size=32)

    def forward(self, x):
        B = x.shape[0]
        # (B, 2, 32, 32) -> (B, 1024, 2)
        x = x.permute(0, 2, 3, 1).reshape(B, 1024, 2)

        x = self.input_proj(x) + self.pos_embedding
        enc = self.encoder(x)

        z = self.up(self.down(enc))
        dec = self.decoder(z, enc)  # (B, 1024, 32)

        # Element-wise output: each token outputs 1 Cholesky parameter
        cholesky_params = self.output_proj(dec).squeeze(-1)  # (B, 1024)

        # Convert to valid density matrix
        out = self.cholesky(cholesky_params)  # (B, 2, 32, 32)

        return out

    def compute_loss(self, pred, target):
        return self.loss_fn(pred, target)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = TransformerCholeskyElementwise()
    print(f"TransformerCholeskyElementwise parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(4, 2, 32, 32)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")

    # Verify output is valid density matrix
    rho = y[:, 0] + 1j * y[:, 1]

    # Check Hermitian
    is_hermitian = torch.allclose(rho, rho.conj().transpose(-2, -1), atol=1e-5)
    print(f"Is Hermitian: {is_hermitian}")

    # Check trace = 1
    traces = torch.diagonal(rho, dim1=-2, dim2=-1).sum(dim=-1)
    print(f"Traces: {traces.real}")

    # Check PSD
    eigvals = torch.linalg.eigvalsh(rho)
    print(f"Min eigenvalue: {eigvals.min().item():.6f} (should be >= 0)")
