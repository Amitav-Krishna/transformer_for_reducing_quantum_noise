import torch
import numpy as np
import os
from scipy.stats import spearmanr, pearsonr

def matrix_sqrt(A):
    """Compute matrix square root via eigendecomposition."""
    # A is (N, N) complex
    eigvals, eigvecs = torch.linalg.eigh(A)
    eigvals = torch.clamp(eigvals, min=0)
    sqrt_eigvals = torch.sqrt(eigvals)
    sqrt_diag = torch.diag_embed(sqrt_eigvals.to(eigvecs.dtype))
    return eigvecs @ sqrt_diag @ eigvecs.mH

def uhlmann_fidelity(rho, sigma):
    """Compute Uhlmann fidelity F(rho, sigma) = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2."""
    rho = (rho + rho.mH) / 2
    sigma = (sigma + sigma.mH) / 2
    
    sqrt_rho = matrix_sqrt(rho)
    inner = sqrt_rho @ sigma @ sqrt_rho
    # Force Hermitian for stability
    inner = (inner + inner.mH) / 2
    sqrt_inner = matrix_sqrt(inner)
    
    trace_val = torch.trace(sqrt_inner).real
    return torch.clamp(trace_val ** 2, 0.0, 1.0)

def frobenius_fidelity(rho, sigma):
    """Compute Frobenius fidelity (cosine similarity)."""
    # Flatten
    rho_flat = rho.flatten()
    sigma_flat = sigma.flatten()
    
    # Inner product
    inner = torch.real(torch.sum(rho_flat.conj() * sigma_flat))
    norm_rho = torch.sqrt(torch.real(torch.sum(rho_flat.conj() * rho_flat)))
    norm_sigma = torch.sqrt(torch.real(torch.sum(sigma_flat.conj() * sigma_flat)))
    
    return inner / (norm_rho * norm_sigma + 1e-12)

def main():
    # Try common locations
    possible_dirs = ["/workspace/dataset_5qubit_float64", "dataset_smaller", "/workspace/dataset_smaller", "dataset", "/workspace/dataset"]
    dataset_dir = None
    for d in possible_dirs:
        if os.path.exists(d):
            dataset_dir = d
            break
            
    if dataset_dir is None:
        print(f"Error: Could not find dataset directory. Checked: {possible_dirs}")
        return

    print(f"Scanning {dataset_dir}...")
    chunk_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith(".pt")])
    print(f"Found {len(chunk_files)} chunks.")
    
    if len(chunk_files) == 0:
        print("No .pt files found.")
        return
    
    frob_scores = []
    uhlmann_scores = []
    
    print("Processing 1 sample per chunk...")
    
    for i, fname in enumerate(chunk_files):
        if i % 100 == 0:
            print(f"  Processed {i}/{len(chunk_files)}")
            
        path = os.path.join(dataset_dir, fname)
        try:
            blob = torch.load(path, map_location="cpu")
            X = blob["X"] # Noisy (B, 32, 32, 2) or (B, 2, 32, 32)
            Y = blob["Y"] # Clean
            
            # Check shape
            # 5-qubit dataset from generate_dataset.py seems to save as (N, 32, 32, 2) numpy then tensor
            # But the loader expects (B, 2, 32, 32)? 
            # generate_dataset.py saves: X.append(np.stack([rho_noisy.real, rho_noisy.imag], axis=-1)) -> (32, 32, 2)
            # Then torch.save saves the tensor.
            # Let's inspect shape of first sample
            
            x0 = X[0]
            y0 = Y[0]
            
            if x0.shape == (32, 32, 2):
                rho_noisy = x0[..., 0] + 1j * x0[..., 1]
            elif x0.shape == (2, 32, 32):
                rho_noisy = x0[0] + 1j * x0[1]
            else:
                print(f"Skipping {fname}: unknown shape {x0.shape}")
                continue
                
            if y0.shape == (32, 32, 2):
                rho_clean = y0[..., 0] + 1j * y0[..., 1]
            elif y0.shape == (2, 32, 32):
                rho_clean = y0[0] + 1j * y0[1]
                
            # Convert to float64/complex128 for precision
            rho_noisy = rho_noisy.to(torch.complex128)
            rho_clean = rho_clean.to(torch.complex128)
            
            # Uhlmann (on unnormalized physical states)
            u_fid = uhlmann_fidelity(rho_noisy, rho_clean).item()
            
            # Frobenius (normalized)
            rho_noisy_norm = rho_noisy / (torch.linalg.norm(rho_noisy) + 1e-12)
            rho_clean_norm = rho_clean / (torch.linalg.norm(rho_clean) + 1e-12)
            f_fid = frobenius_fidelity(rho_noisy_norm, rho_clean_norm).item()
            
            frob_scores.append(f_fid)
            uhlmann_scores.append(u_fid)
            
        except Exception as e:
            print(f"Error reading {fname}: {e}")
            continue
            
    frob_scores = np.array(frob_scores)
    uhlmann_scores = np.array(uhlmann_scores)
    
    print(f"Computed scores for {len(frob_scores)} samples.")
    
    if len(frob_scores) > 1:
        spearman, _ = spearmanr(frob_scores, uhlmann_scores)
        pearson, _ = pearsonr(frob_scores, uhlmann_scores)
        
        print(f"Spearman Rank Correlation: {spearman:.4f}")
        print(f"Pearson Correlation: {pearson:.4f}")

        # Generate plot
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(8, 6))
            plt.scatter(frob_scores, uhlmann_scores, alpha=0.6, edgecolors='none', s=20)
            
            # Add trend line
            z = np.polyfit(frob_scores, uhlmann_scores, 1)
            p = np.poly1d(z)
            plt.plot(np.sort(frob_scores), p(np.sort(frob_scores)), "r--", alpha=0.8, 
                     label=f"Trend (r={pearson:.2f})")
            
            plt.title(f"Frobenius vs Uhlmann Fidelity (5-qubit subset, N={len(frob_scores)})")
            plt.xlabel("Frobenius Fidelity (Normalized)")
            plt.ylabel("Uhlmann Fidelity (Physical)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save
            out_path = "/workspace/frob_vs_uhlmann_correlation.pdf"
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {out_path}")
            
        except ImportError:
            print("matplotlib not installed, skipping plot generation")
        except Exception as e:
            print(f"Error generating plot: {e}")

    else:
        print("Not enough samples for correlation.")

if __name__ == "__main__":
    main()

