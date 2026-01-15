#!/usr/bin/env python3
"""
Uhlmann fidelity evaluation for 5-qubit ablation models (Deep/Wide).
"""

import sys
import torch
import os
from pathlib import Path
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training_loop.dataset.ChunkDataset import ChunkDataset
from training_loop.dataset.load_chunks import load_chunks
from training_loop.dataset.split_chunks import split_chunks

from train_16.models.mlp_5qubit_deep import HierarchicalMLP5QubitDeep
from train_16.models.mlp_5qubit_wide import HierarchicalMLP5QubitWide

DTYPE = torch.float64
CDTYPE = torch.complex128

def normalize_to_unit(x):
    """Normalize tensor to unit Frobenius norm."""
    norm = x.flatten().norm() + 1e-8
    return x / norm

def matrix_sqrt_float64(A):
    A = A.to(CDTYPE)
    eigvals, eigvecs = torch.linalg.eigh(A)
    if torch.isnan(eigvals).any() or torch.isinf(eigvals).any():
        # Fallback for numerical stability
        return matrix_sqrt_safe(A)
    eigvals = torch.clamp(eigvals.real, min=0.0)
    sqrt_eigvals = torch.sqrt(eigvals)
    sqrt_diag = torch.diag(sqrt_eigvals.to(CDTYPE))
    return eigvecs @ sqrt_diag @ eigvecs.mH

def matrix_sqrt_safe(A):
    """Slower but safer matrix sqrt using SVD if eigh fails."""
    try:
        u, s, vh = torch.linalg.svd(A)
        s_sqrt = torch.sqrt(torch.clamp(s, min=0.0))
        return u @ torch.diag(s_sqrt.to(CDTYPE)) @ vh
    except:
        return torch.eye(A.shape[-1], device=A.device, dtype=CDTYPE)

def uhlmann_fidelity_single(rho, sigma):
    rho = (rho + rho.mH) / 2
    sigma = (sigma + sigma.mH) / 2
    rho = rho / (torch.trace(rho).real + 1e-12)
    sigma = sigma / (torch.trace(sigma).real + 1e-12)
    
    sqrt_rho = matrix_sqrt_float64(rho)
    inner = sqrt_rho @ sigma @ sqrt_rho
    inner = (inner + inner.mH) / 2
    sqrt_inner = matrix_sqrt_float64(inner)
    trace_val = torch.trace(sqrt_inner).real
    fid = (trace_val**2).item()
    return max(0.0, min(1.0, fid))

def tensor_to_complex(t):
    return t[0].to(CDTYPE) + 1j * t[1].to(CDTYPE)

@torch.no_grad()
def evaluate_model_fidelity(model, chunks, device, batch_size=64):
    model.eval()
    all_fidelities = []
    n_failures = 0
    
    # Process only first 500 samples to save time if needed, 
    # but for full eval we should do all.
    # Let's do full test set as it's small enough (10k samples).
    
    for X, Y in chunks:
        ds = ChunkDataset(X, Y, mode="transformer")
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        
        for x, y in loader:
            x = x.to(device, dtype=DTYPE)
            y = y.to(device, dtype=DTYPE)
            
            # Normalize inputs
            x_norm = torch.stack([normalize_to_unit(xi) for xi in x])
            
            pred = model(x_norm)
            
            for i in range(pred.shape[0]):
                pred_mat = tensor_to_complex(pred[i])
                clean_mat = tensor_to_complex(y[i])
                
                try:
                    fid = uhlmann_fidelity_single(pred_mat, clean_mat)
                    all_fidelities.append(fid)
                except Exception:
                    n_failures += 1
                    
    fids = torch.tensor(all_fidelities, dtype=DTYPE)
    return {
        "mean": fids.mean().item(),
        "std": fids.std().item(),
        "count": len(all_fidelities),
        "failures": n_failures
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python eval_ablation.py [deep|wide]")
        sys.exit(1)
        
    mode = sys.argv[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load data
    print("Loading dataset...")
    chunks = load_chunks("dataset_5qubit_float64")
    _, _, test_chunks = split_chunks(chunks, 0.8, 0.1, seed=42)
    print(f"Test chunks: {len(test_chunks)}")
    
    # Configure model
    if mode == "deep":
        print("\nEvaluating Deep MLP (8 layers)")
        model = HierarchicalMLP5QubitDeep(loss_fn=None).to(device)
        ckpt_path = "train_17/checkpoints_17/mlp_deep/best.pt"
    elif mode == "wide":
        print("\nEvaluating Wide MLP (2048 hidden)")
        model = HierarchicalMLP5QubitWide(loss_fn=None).to(device)
        ckpt_path = "train_17/checkpoints_17/mlp_wide/best.pt"
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
        
    print(f"Loading checkpoint: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        print("Checkpoint not found!")
        sys.exit(1)
        
    # We need to map location because we might be on a different GPU
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    
    res = evaluate_model_fidelity(model, test_chunks, device)
    
    print("\n" + "="*50)
    print(f"RESULTS: {mode.upper()} MLP")
    print("="*50)
    print(f"Mean Fidelity: {res['mean']:.6f}")
    print(f"Std Dev:       {res['std']:.6f}")
    print(f"Samples:       {res['count']}")
    print(f"Failures:      {res['failures']}")
    print("="*50)

if __name__ == "__main__":
    main()

