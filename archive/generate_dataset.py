from sklearn.model_selection import train_test_split
import cirq
import numpy as np
import random
import torch
from tqdm import tqdm
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

SINGLE_QUBIT_GATES = ["X", "Y", "Z", "H", "T", "S", "rx", "ry", "rz"]
TWO_QUBIT_GATES = ["cx", "cz", "swap"]



def generate_random_circuit(num_qubits, depth, seed=None):
    """
    Generate a random quantum circuit with specified number of qubits and depth.

    Args:
        num_qubits: Number of qubits in the circuit
        depth: Number of layers of gates
        seed: Random seed for reproducibility

    Returns:
        A random Cirq circuit
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    qubits = cirq.LineQubit.range(num_qubits)
    circuit = cirq.Circuit()

    # Gates to sample from
    single_qubit_gates = [
        cirq.X,
        cirq.Y,
        cirq.Z,
        cirq.H,
        cirq.T,
        cirq.S,
        lambda q: cirq.rx(np.random.uniform(0, 2 * np.pi))(q),
        lambda q: cirq.ry(np.random.uniform(0, 2 * np.pi))(q),
        lambda q: cirq.rz(np.random.uniform(0, 2 * np.pi))(q),
    ]

    two_qubit_gates = [
        cirq.CZ,
        cirq.CNOT,
        lambda q1, q2: cirq.SWAP(q1, q2),
    ]

    for d in range(depth):
        # Add single-qubit gates
        for q in qubits:
            gate = random.choice(single_qubit_gates)
            circuit.append(gate(q))

        # Add two-qubit gates (to create entanglement)
        qubit_pairs = list(zip(qubits[:-1], qubits[1:]))  # Adjacent qubits
        for q1, q2 in random.sample(
            qubit_pairs, k=min(len(qubit_pairs), num_qubits // 2)
        ):
            gate = random.choice(two_qubit_gates)
            circuit.append(gate(q1, q2))

    return circuit


# In[6]:


def add_noise(circuit, noise_type, noise_level):
    """
    Add specified noise model to a quantum circuit.

    Args:
        circuit: The original quantum circuit
        noise_type: Type of noise ('depolarizing', 'amplitude_damping', 'phase_damping', 'bitflip', 'mixed')
        noise_level: The strength of the noise (0 to 1)

    Returns:
        A noisy quantum circuit
    """
    qubits = sorted(circuit.all_qubits())
    noisy_circuit = cirq.Circuit()

    if noise_type == "depolarizing":
        # Depolarizing noise: replaces the qubit state with a completely mixed state with probability p
        noise_model = cirq.depolarize(p=noise_level)
        for moment in circuit:
            noisy_circuit.append(moment)
            noisy_circuit.append(noise_model.on_each(*qubits))
    elif noise_type == "amplitude_damping":
        # Amplitude damping: models energy dissipation (e.g., spontaneous emission)
        noise_model = cirq.amplitude_damp(gamma=noise_level)
        for moment in circuit:
            noisy_circuit.append(moment)
            noisy_circuit.append(noise_model.on_each(*qubits))
    elif noise_type == "phase_damping":
        # Phase damping: models dephasing (loss of coherence without energy loss)
        noise_model = cirq.phase_damp(gamma=noise_level)
        for moment in circuit:
            noisy_circuit.append(moment)
            noisy_circuit.append(noise_model.on_each(*qubits))
    elif noise_type == "bitflip":
        # Bit-flip noise: flips the qubit state with probability p
        noise_model = cirq.bit_flip(p=noise_level)
        for moment in circuit:
            noisy_circuit.append(moment)
            noisy_circuit.append(noise_model.on_each(*qubits))
    elif noise_type == "mixed":
        # Apply a mix of different noise types with the given level
        # Apply each noise channel individually
        noise_channels = [
            cirq.depolarize(p=noise_level / 4),
            cirq.amplitude_damp(gamma=noise_level / 4),
            cirq.phase_damp(gamma=noise_level / 4),
            cirq.bit_flip(p=noise_level / 4),
        ]
        for moment in circuit:
            noisy_circuit.append(moment)
            for channel in noise_channels:
                noisy_circuit.append(channel.on_each(*qubits))
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    return noisy_circuit


# In[7]:


def simulate_density_matrix(circuit, qubits):
    simulator = cirq.DensityMatrixSimulator()
    result = simulator.simulate(circuit, qubit_order=qubits)
    return result.final_density_matrix


# In[8]:


def normalize_by_frobenius(X):
    """
    Normalize each (32,32,2) matrix by its Frobenius norm.
    X: np.ndarray of shape (N, H, W, 2)
    """
    norms = np.sqrt(np.sum(X[..., 0]**2 + X[..., 1]**2, axis=(1,2), keepdims=True))
    X_norm = X / (norms[..., None] + 1e-12)
    return X_norm


# In[9]:


def generate_dataset(
    num_samples_per_cell=100,
    num_qubits=5,
    min_depth=6,
    max_depth=9,
    noise_types=None,
    noise_levels=None,
    seed=None,
):
    """
    Generate dataset of noisy and clean quantum density matrices.

    Args:
        num_samples_per_cell: Number of samples for each combination of noise type and level.
        num_qubits: Number of qubits for the circuits.
        min_depth: Minimum depth for random circuits.
        max_depth: Maximum depth for random circuits.
        noise_types: List of noise types to include (defaults to common types).
        noise_levels: List of noise levels to include (defaults to common levels).
        seed: Random seed for reproducibility.

    Returns:
        Tuple containing:
            - X: Noisy density matrices (NumPy array, shape: N x 2^n x 2^n x 2, last dim for real/imag)
            - Y: Clean density matrices (NumPy array, shape: N x 2^n x 2^n x 2, last dim for real/imag)
            - meta: List of dictionaries with metadata for each sample.
    """
    if noise_types is None:
        noise_types = ["depolarizing", "amplitude_damping", "phase_damping", "bitflip", "mixed"]
    if noise_levels is None:
        noise_levels = [0.05, 0.10, 0.15, 0.20]

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    X, Y, meta = [], [], []
    total_combos = len(noise_types) * len(noise_levels)
    total_samples = num_samples_per_cell * total_combos

    print(
        f"Generating {total_samples} samples "
        f"({len(noise_types)} noise types × {len(noise_levels)} levels)..."
    )

    simulator = cirq.DensityMatrixSimulator()

    with tqdm(total=total_samples) as pbar:
        for noise_type in noise_types:
            for noise_level in noise_levels:
                for i in range(num_samples_per_cell):
                    print("Sample:", i + noise_levels.index(noise_level) + noise_types.index(noise_type))
                    depth = random.randint(min_depth, max_depth)
                    circuit = generate_random_circuit(num_qubits, depth)
                    qubits = sorted(circuit.all_qubits())

                    # Simulate clean density matrix
                    result_clean = simulator.simulate(circuit, qubit_order=qubits)
                    rho_clean = result_clean.final_density_matrix

                    # Add noise and simulate noisy density matrix
                    noisy_circuit = add_noise(circuit, noise_type, noise_level)
                    result_noisy = simulator.simulate(noisy_circuit, qubit_order=qubits)
                    rho_noisy = result_noisy.final_density_matrix

                    # Stack real/imag parts for ML input
                    X.append(np.stack([rho_noisy.real, rho_noisy.imag], axis=-1))
                    Y.append(np.stack([rho_clean.real, rho_clean.imag], axis=-1))
                    meta.append(
                        {
                            "noise_type": noise_type,
                            "noise_level": noise_level,
                            "depth": depth,
                        }
                    )
                    pbar.update(1)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    print(f"✅ Dataset shape: X={X.shape}, Y={Y.shape}")
    return X, Y, meta


# In[10]:


def split_dataset(X, Y, meta, test_size=0.2, random_state=42):
    """
    Split dataset into train/test while keeping equal representation
    across (noise_type, noise_level) cells.
    """

    # Turn metadata into a structured label for stratification
    meta_labels = [f"{m['noise_type']}_{m['noise_level']}" for m in meta]

    # Use sklearn stratified split to preserve proportions per cell
    idx_train, idx_test = train_test_split(
        np.arange(len(X)),
        test_size=test_size,
        random_state=random_state,
        stratify=meta_labels
    )

    X_train, X_test = X[idx_train], X[idx_test]
    Y_train, Y_test = Y[idx_train], Y[idx_test]

    meta_train = [meta[i] for i in idx_train]
    meta_test  = [meta[i] for i in idx_test]

    print(f"Train: {len(X_train)}  Test: {len(X_test)}")
    return (X_train, Y_train, meta_train), (X_test, Y_test, meta_test)


# # Transformer architecture
# 
# Papers to reference
# 
# @misc{kendre2025machinelearningquantumnoise,
#       title={Machine Learning for Quantum Noise Reduction},
#       author={Karan Kendre},
#       year={2025},
#       eprint={2509.16242},
#       archivePrefix={arXiv},
#       primaryClass={quant-ph},
#       url={https://arxiv.org/abs/2509.16242},
#       file="./2509.16242v1.pdf",
# }
# 
# @misc{norambuena2023physicsinformedneuralnetworksquantum,
#       title={Physics-informed neural networks for quantum control},
#       author={Ariel Norambuena and Marios Mattheakis and Francisco J. González and Raúl Coto},
#       year={2023},
#       eprint={2206.06287},
#       archivePrefix={arXiv},
#       primaryClass={quant-ph},
#       url={https://arxiv.org/abs/2206.06287},
#       file="./2206.06287v2.pdf",
# }

# In[ ]:





# In[11]:


class QuantumDenoisingDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx].permute(2, 0, 1), self.Y[idx].permute(2, 0, 1)


# In[12]:


class PatchEmbed(nn.Module):
    def __init__(self, in_chans=2, embed_dim=64, patch_size=2):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1,2)
        return x, (H, W)


# In[13]:


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


# In[14]:


class QuantumTransformerDenoiser(nn.Module):
    def __init__(self, patch_size=2, embed_dim=64, depth=4, num_heads=4):
        super().__init__()
        self.embed = PatchEmbed(in_chans=2, embed_dim=embed_dim, patch_size=patch_size)
        self.blocks = nn.Sequential(*[TransformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.deproj = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 2, kernel_size=patch_size, stride=patch_size)
        )

    def forward(self, x):
        # x: (B, 2, H, W)
        tokens, (H, W) = self.embed(x)
        tokens = self.blocks(tokens)
        # Reconstruct spatial map
        B, N, C = tokens.shape
        tokens = tokens.transpose(1,2).view(B, C, H, W)
        A = self.deproj(tokens)
        rho_corr = x + A
        rho=project_to_density(rho_corr)
        return rho  # (B, 2, H_orig, W_orig)


# In[15]:


def project_to_density(A):
    """
    Convert A (2-channel tensor: real, imag) into a valid density matrix
    ρ = AA† / Tr(AA†), batched over the leading dimension.
    """
    # Complex reconstruction
    A_complex = A[:,0] + 1j*A[:,1]  # (B, H, W)
    rho = A_complex @ A_complex.conj().transpose(-1, -2)

    # Compute trace for each batch safely
    tr = torch.real(torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1)).clamp(min=1e-12)
    rho = rho / tr.view(-1, 1, 1)

    # Split back into two channels
    rho_re, rho_im = rho.real, rho.imag
    return torch.stack([rho_re, rho_im], dim=1)


# In[16]:


def fidelity_loss(pred, target):
    # pred,target: (B, 2, H, W)
    pred_c = pred[:,0] + 1j * pred[:,1]
    targ_c = target[:,0] + 1j * target[:,1]
    # Frobenius inner product
    num = torch.real(torch.sum(pred_c.conj() * targ_c, dim=[1,2]))
    denom = torch.sqrt(torch.sum(torch.abs(pred_c)**2, dim=[1,2]) * torch.sum(torch.abs(targ_c)**2, dim=[1,2]))
    fid = num / (denom + 1e-8)
    return 1 - torch.mean(fid)


# In[17]:


def total_loss(pred, target, λ_herm=1.0, λ_trace=1.0, λ_psd=1.0, α=0.5, eps=1e-8, jitter=1e-6, reduce=True):
    """
    Physics-informed composite loss for quantum density matrix regression.
    """

    # --- Convert to complex tensors ---
    pred_c = pred[:, 0] + 1j * pred[:, 1]
    targ_c = target[:, 0] + 1j * target[:, 1]

    # --- 1. Reconstruction (MSE + Fidelity) ---b
    # Calculate per-sample MSE
    L_mse_per_sample = F.mse_loss(pred, target, reduction='none').sum(dim=[1, 2, 3])
    # Calculate per-sample Fidelity
    num = torch.real(torch.sum(pred_c.conj() * targ_c, dim=(1, 2)))
    denom = torch.sqrt(
        torch.sum(torch.abs(pred_c)**2, dim=(1, 2)) *
        torch.sum(torch.abs(targ_c)**2, dim=(1, 2))
    ) + eps
    L_fid_per_sample = 1 - (num / denom)
    L_rec_per_sample = α * L_mse_per_sample + (1 - α) * L_fid_per_sample


    # --- 2. Hermiticity ---
    herm_diff = pred_c - pred_c.conj().transpose(-1, -2)
    L_herm_per_sample = torch.sum(torch.abs(herm_diff)**2, dim=(1, 2))

    # --- 3. Trace normalization ---
    tr = torch.real(torch.diagonal(pred_c, dim1=-2, dim2=-1).sum(-1))
    L_trace_per_sample = (tr - 1)**2

    # --- 4. Positive semidefiniteness ---
    # Stabilize: symmetrize and add jitter to diagonal
    sym = (pred_c + pred_c.conj().transpose(-1, -2)) / 2
    sym = sym + jitter * torch.eye(sym.shape[-1], device=sym.device).unsqueeze(0)
    try:
        eigvals = torch.linalg.eigvalsh(sym)
    except torch._C._LinAlgError:
        # fallback: detach and compute with numpy if a single batch element fails
        eigvals = torch.tensor(
            np.linalg.eigvalsh(sym.detach().cpu().numpy()),
            device=sym.device, dtype=torch.float32
        )

    L_psd_per_sample = F.softplus(-eigvals).mean(dim=-1) # Mean over eigenvalues for each sample


    # --- Combine ---
    total_per_sample = L_rec_per_sample + λ_herm * L_herm_per_sample + λ_trace * L_trace_per_sample + λ_psd * L_psd_per_sample

    if reduce:
        return torch.mean(total_per_sample)
    else:
        return total_per_sample
# In[2342]:
import os
import numpy as np
import torch
import cirq
import random
from tqdm import tqdm

def generate_dataset_streaming(
    output_dir="dataset",
    num_samples_per_cell=100,
    num_qubits=5,
    min_depth=6,
    max_depth=9,
    noise_types=None,
    noise_levels=None,
    chunk_size=1000,
    seed=None,
    save_format="npz",  # options: "npz" or "pt"
):
    """
    Generate a large quantum denoising dataset and stream it to disk chunk-by-chunk.

    Each chunk file will contain up to `chunk_size` samples and be named:
        dataset/{noise_type}_{noise_level}_part{k}.{npz|pt}

    Args:
        output_dir: Directory to save the chunk files.
        num_samples_per_cell: Number of samples for each (noise_type, noise_level).
        num_qubits: Number of qubits per circuit.
        min_depth, max_depth: Range of random circuit depths.
        noise_types, noise_levels: Lists of noise models and their levels.
        chunk_size: Number of samples to accumulate before saving a chunk.
        seed: Optional random seed for reproducibility.
        save_format: 'npz' (NumPy compressed) or 'pt' (PyTorch tensor dict)
    """
    os.makedirs(output_dir, exist_ok=True)

    if noise_types is None:
        noise_types = ["depolarizing", "amplitude_damping", "phase_damping", "bitflip", "mixed"]
    if noise_levels is None:
        noise_levels = [0.05, 0.10, 0.15, 0.20]

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    simulator = cirq.DensityMatrixSimulator()
    total_combos = len(noise_types) * len(noise_levels)
    total_samples = num_samples_per_cell * total_combos

    print(f"Streaming {total_samples} samples to '{output_dir}'")

    sample_counter = 0
    chunk_counter = 0
    X_chunk, Y_chunk, meta_chunk = [], [], []

    with tqdm(total=total_samples) as pbar:
        for noise_type in noise_types:
            for noise_level in noise_levels:
                for i in range(num_samples_per_cell):
                    depth = random.randint(min_depth, max_depth)
                    circuit = generate_random_circuit(num_qubits, depth)
                    qubits = sorted(circuit.all_qubits())

                    # Clean and noisy simulations
                    rho_clean = simulator.simulate(circuit, qubit_order=qubits).final_density_matrix
                    noisy_circuit = add_noise(circuit, noise_type, noise_level)
                    rho_noisy = simulator.simulate(noisy_circuit, qubit_order=qubits).final_density_matrix

                    X_chunk.append(np.stack([rho_noisy.real, rho_noisy.imag], axis=-1))
                    Y_chunk.append(np.stack([rho_clean.real, rho_clean.imag], axis=-1))
                    meta_chunk.append({
                        "noise_type": noise_type,
                        "noise_level": noise_level,
                        "depth": depth,
                    })

                    sample_counter += 1
                    pbar.update(1)

                    # Save when chunk full
                    if len(X_chunk) >= chunk_size:
                        chunk_counter += 1
                        fname = f"{output_dir}/{noise_type}_{noise_level}_part{chunk_counter}.{save_format}"

                        if save_format == "npz":
                            np.savez_compressed(fname, X=np.array(X_chunk, dtype=np.float32),
                                                Y=np.array(Y_chunk, dtype=np.float32),
                                                meta=np.array(meta_chunk, dtype=object))
                        else:
                            torch.save({
                                "X": torch.tensor(X_chunk, dtype=torch.float32),
                                "Y": torch.tensor(Y_chunk, dtype=torch.float32),
                                "meta": meta_chunk,
                            }, fname)

                        print(f"💾 Saved chunk {chunk_counter}: {fname} ({len(X_chunk)} samples)")
                        X_chunk, Y_chunk, meta_chunk = [], [], []  # free memory

    # Save final remainder
    if len(X_chunk) > 0:
        chunk_counter += 1
        fname = f"{output_dir}/final_part{chunk_counter}.{save_format}"
        if save_format == "npz":
            np.savez_compressed(fname, X=np.array(X_chunk, dtype=np.float32),
                                Y=np.array(Y_chunk, dtype=np.float32),
                                meta=np.array(meta_chunk, dtype=object))
        else:
            torch.save({
                "X": torch.tensor(X_chunk, dtype=torch.float32),
                "Y": torch.tensor(Y_chunk, dtype=torch.float32),
                "meta": meta_chunk,
            }, fname)
        print(f"💾 Saved final chunk: {fname} ({len(X_chunk)} samples)")

    print(f"✅ Finished streaming dataset: {sample_counter} samples, {chunk_counter} chunks total.")


# In[18]:


import torch
import torch.nn.functional as F

def simple_loss(pred, target, α=0.5, eps=1e-8, reduce=True):
    """
    Simple baseline loss for quantum denoising (no physics constraints).
    Combines MSE and fidelity only.
    """
    pred_c = pred[:, 0] + 1j * pred[:, 1]
    targ_c = target[:, 0] + 1j * target[:, 1]

    # Mean squared error
    L_mse_per_sample = F.mse_loss(pred, target, reduction='none').sum(dim=[1, 2, 3])

    # Fidelity term
    num = torch.real(torch.sum(pred_c.conj() * targ_c, dim=(1, 2)))
    denom = torch.sqrt(
        torch.sum(torch.abs(pred_c)**2, dim=(1, 2)) *
        torch.sum(torch.abs(targ_c)**2, dim=(1, 2))
    ) + eps
    L_fid_per_sample = 1 - (num / denom)

    # Combine
    total_per_sample = α * L_mse_per_sample + (1 - α) * L_fid_per_sample
    return total_per_sample.mean() if reduce else total_per_sample


# # Training (Run everything up to here)

# In[ ]:


generate_dataset_streaming(
    output_dir="dataset",
    num_samples_per_cell=50000,   # 5 noise types × 4 levels × 50000 = 1,000,000 total
    num_qubits=5,
    chunk_size=10,
    save_format="pt"
)
