import torch
import numpy as np
from scipy.stats import spearmanr
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_loop.dataset.load_chunks import load_chunks_with_metadata
from training_loop.dataset.split_chunks import split_chunks


@torch.no_grad()
def matrix_sqrt(m):
    evals, evecs = torch.linalg.eigh(m)
    evals = torch.clamp(evals, min=0)
    return (evecs * torch.sqrt(evals)[..., None, :]) @ evecs.conj().transpose(-1, -2)


@torch.no_grad()
def uhlmann_fidelity(rho, sigma):
    rho_c = rho[..., 0] + 1j * rho[..., 1]
    sig_c = sigma[..., 0] + 1j * sigma[..., 1]
    sr = matrix_sqrt(rho_c)
    inner = sr @ sig_c @ sr
    s_inner = matrix_sqrt(inner)
    tr = torch.real(torch.diagonal(s_inner, dim1=-2, dim2=-1).sum(-1))
    return tr**2


@torch.no_grad()
def frobenius_fidelity(rho, sigma, eps=1e-8):
    a = rho[..., 0] + 1j * rho[..., 1]
    b = sigma[..., 0] + 1j * sigma[..., 1]
    num = torch.real(torch.sum(a.conj() * b, dim=(-1, -2)))
    denom = torch.sqrt(
        torch.sum(torch.abs(a)**2, dim=(-1, -2)) *
        torch.sum(torch.abs(b)**2, dim=(-1, -2)) + eps
    )
    return torch.clamp(num / denom, -1, 1)


def main():
    print("Loading dataset...")
    chunks = load_chunks_with_metadata("dataset_smaller")
    _, _, test_chunks = split_chunks(chunks, seed=42)

    all_uhlmann = []
    all_frobenius = []

    print("Computing fidelities...")
    for blob in test_chunks:
        X = blob["X"].to(dtype=torch.float64)
        Y = blob["Y"].to(dtype=torch.float64)

        uhl = uhlmann_fidelity(X, Y).cpu().numpy()
        frob = frobenius_fidelity(X, Y).cpu().numpy()

        all_uhlmann.extend(uhl.tolist())
        all_frobenius.extend(frob.tolist())

    all_uhlmann = np.array(all_uhlmann)
    all_frobenius = np.array(all_frobenius)

    rho, pval = spearmanr(all_uhlmann, all_frobenius)

    print(f"\nN = {len(all_uhlmann)} test samples")
    print(f"Spearman's rho = {rho:.6f}")
    print(f"p-value = {pval:.2e}")


if __name__ == "__main__":
    main()