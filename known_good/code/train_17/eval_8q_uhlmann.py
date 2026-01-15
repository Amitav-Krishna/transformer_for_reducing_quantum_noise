import torch
import sys
import os
from pathlib import Path
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from training_loop.dataset.StreamingChunkDataset import get_chunk_files, split_chunk_files, StreamingChunkDataset
from train_16.models.transformer_8qubit import HierarchicalTransformer8Qubit
from train_16.models.mlp_8qubit import HierarchicalMLP8Qubit

DTYPE = torch.float64
CDTYPE = torch.complex128

def normalize_to_unit(x):
    """Normalize tensor to unit Frobenius norm."""
    norm = x.flatten().norm() + 1e-8
    return x / norm

def matrix_sqrt_float64(A):
    A = A.to(CDTYPE)
    eigvals, eigvecs = torch.linalg.eigh(A)
    # Clamp small negative eigenvalues (numerical errors)
    eigvals = torch.clamp(eigvals.real, min=0.0)
    sqrt_eigvals = torch.sqrt(eigvals)
    sqrt_diag = torch.diag(sqrt_eigvals.to(CDTYPE))
    return eigvecs @ sqrt_diag @ eigvecs.mH

def uhlmann_fidelity_single(rho, sigma):
    """
    Compute Uhlmann fidelity for a single pair of density matrices.
    """
    # Ensure Hermitian
    rho = (rho + rho.mH) / 2
    sigma = (sigma + sigma.mH) / 2
    
    # Normalize trace to 1 (valid density matrix)
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
def evaluate_fidelity_streaming(model, chunk_files, device, limit_batches=100):
    """
    Evaluate Uhlmann fidelity on streaming dataset.
    
    Args:
        model: Loaded model (or None for baseline)
        chunk_files: List of chunk file paths
        device: Torch device
        limit_batches: Limit number of batches to evaluate (to save time)
    """
    if model:
        model.eval()
        
    all_fidelities = []
    n_failures = 0
    
    # Process files
    processed_batches = 0
    
    for fpath in chunk_files:
        if processed_batches >= limit_batches:
            break
            
        try:
            blob = torch.load(fpath, weights_only=False, map_location="cpu")
            X = blob["X"].double() # (N, 2, 256, 256)
            Y = blob["Y"].double() # (N, 2, 256, 256)
            
            # Create batches manually
            batch_size = 4
            for i in range(0, len(X), batch_size):
                if processed_batches >= limit_batches:
                    break
                    
                x_batch = X[i:i+batch_size].permute(0, 3, 1, 2) # (B, 2, 256, 256)
                y_batch = Y[i:i+batch_size].permute(0, 3, 1, 2)
                
                # Move to device
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
                if model:
                    # Normalize inputs for model
                    x_norm = torch.stack([normalize_to_unit(xi) for xi in x_batch])
                    pred = model(x_norm)
                else:
                    # Baseline: prediction is just the noisy input
                    pred = x_batch
                
                # Compute fidelity for batch
                for j in range(len(pred)):
                    p_mat = tensor_to_complex(pred[j])
                    c_mat = tensor_to_complex(y_batch[j])
                    
                    try:
                        fid = uhlmann_fidelity_single(p_mat, c_mat)
                        all_fidelities.append(fid)
                    except Exception as e:
                        n_failures += 1
                
                processed_batches += 1
                if processed_batches % 10 == 0:
                    print(f"Processed {processed_batches} batches...")
                    
        except Exception as e:
            print(f"Error processing chunk {fpath}: {e}")
            continue
            
    if not all_fidelities:
        return {"mean": 0.0, "std": 0.0, "count": 0}
        
    fids = torch.tensor(all_fidelities)
    return {
        "mean": fids.mean().item(),
        "std": fids.std().item(),
        "count": len(all_fidelities),
        "failures": n_failures
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python eval_8q_uhlmann.py [transformer|mlp]")
        sys.exit(1)
        
    model_type = sys.argv[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get test files
    base_dir = "/workspace"
    if not os.path.exists(base_dir):
         # Fallback for local testing
         base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    chunk_files = get_chunk_files("dataset_8qubit_float64")
    _, _, test_files = split_chunk_files(chunk_files)
    print(f"Found {len(test_files)} test chunks")
    
    # Evaluate Baseline first
    print("\n=== Evaluating Baseline (Noisy vs Clean) ===")
    base_res = evaluate_fidelity_streaming(None, test_files, device, limit_batches=50)
    print(f"Baseline Fidelity: {base_res['mean']:.6f} ± {base_res['std']:.6f}")
    
    # Evaluate Model
    print(f"\n=== Evaluating {model_type.upper()} ===")
    
    if model_type == "transformer":
        model = HierarchicalTransformer8Qubit(loss_fn=None).to(device)
        ckpt_path = os.path.join(base_dir, "train_17/checkpoints_17/transformer_8q/best.pt")
    elif model_type == "mlp":
        model = HierarchicalMLP8Qubit(loss_fn=None).to(device)
        ckpt_path = os.path.join(base_dir, "train_17/checkpoints_17/mlp_8q/best.pt")
    else:
        print(f"Unknown model type: {model_type}")
        sys.exit(1)
        
    print(f"Loading checkpoint from {ckpt_path}")
    if not os.path.exists(ckpt_path):
        print("Checkpoint not found!")
        sys.exit(1)
        
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    
    res = evaluate_fidelity_streaming(model, test_files, device, limit_batches=100)
    print(f"{model_type.upper()} Fidelity: {res['mean']:.6f} ± {res['std']:.6f}")
    print(f"Improvement: {res['mean']/base_res['mean']:.2f}x")

if __name__ == "__main__":
    main()

