import torch
import numpy as np
from training_loop.dataset.load_chunks import load_chunks_with_metadata
from training_loop.dataset.split_chunks import split_chunks
import pandas as pd
import os


@torch.no_grad()
def matrix_sqrt(m):
    # m: (..., N, N) complex Hermitian
    evals, evecs = torch.linalg.eigh(m)
    evals = torch.clamp(evals, min=0)
    return (evecs * torch.sqrt(evals)[..., None, :]) @ evecs.conj().transpose(-1, -2)


@torch.no_grad()
def uhlmann_fidelity(rho, sigma):
    """
    rho, sigma: (B, H, W, 2) real/imag channels in last dim
    returns (B,) fidelities
    """
    # convert to complex
    rho_c = rho[..., 0] + 1j * rho[..., 1]
    sig_c = sigma[..., 0] + 1j * sigma[..., 1]

    # sqrt(rho)
    sr = matrix_sqrt(rho_c)
    inner = sr @ sig_c @ sr

    # sqrt(inner)
    s_inner = matrix_sqrt(inner)

    # fidelity = (Tr sqrt( sqrt(rho) sigma sqrt(rho) ))^2
    tr = torch.real(torch.diagonal(s_inner, dim1=-2, dim2=-1).sum(-1))
    return tr**2


def main():
    print("Loading dataset...")
    chunks = load_chunks_with_metadata("dataset_smaller")
    _, _, test_chunks = split_chunks(chunks, seed=42)

    all_values = []

    print("Evaluating Uhlmann fidelity for noisy → clean test states...")
    for blob in test_chunks:
        X = blob["X"]  # noisy
        Y = blob["Y"]  # clean
        meta = blob["meta"]

        # tensorify (use .to() to avoid warning when already a tensor)
        X = X.detach().clone().to(dtype=torch.float64)
        Y = Y.detach().clone().to(dtype=torch.float64)

        # compute samplewise fidelity
        fids = uhlmann_fidelity(X, Y).cpu().numpy()

        for f, m in zip(fids, meta):
            all_values.append({
                "noise_type": m["noise_type"],
                "noise_level": m["noise_level"],
                "uhlmann_fidelity": float(f)
            })

    df = pd.DataFrame(all_values)
    os.makedirs("csvs_2/uhlmann_ground_truth", exist_ok=True)
    out_path = "csvs_2/uhlmann_ground_truth/noisy_vs_clean_test_uhlmann.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved Uhlmann fidelities to {out_path}")
    print(df.groupby(["noise_type", "noise_level"]).mean())


if __name__ == "__main__":
    main()
