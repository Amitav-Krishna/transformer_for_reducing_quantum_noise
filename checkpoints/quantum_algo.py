#!/usr/bin/env python
# coding: utf-8

# # Initialization

# In[7]:


get_ipython().system('pip install numpy qiskit torch matplotlib tqdm torchvision cirq')


# In[41]:
print("Platypus")
# In[8]:


import numpy as np
import cirq
import random

SINGLE_QUBIT_GATES = ["X", "Y", "Z", "H", "T", "S", "rx", "ry", "rz"]
TWO_QUBIT_GATES = ["cx", "cz", "swap"]


# # Quantum circuit simulations

# In[9]:


def random_circuit(n_qubits: int, depth: float) -> tuple[cirq.Circuit, list[cirq.LineQubit]]:
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    for _ in range(depth):
        if random.random() < 0.7:
            q = random.choice(qubits)
            g = random.choice(SINGLE_QUBIT_GATES)
            if g in ["rx", "ry", "rz"]:
                theta = 2*np.pi*random.random()
                circuit.append(getattr(cirq, g)(theta)(q))
            else: 
                circuit.append(getattr(cirq, g)(q))
        else:
            q1, q2 = random.sample(qubits, 2)
            g = random.choice(TWO_QUBIT_GATES)
            if g == "cx":
                circuit.append(cirq.CNOT(q1, q2))
            elif g == "cz":
                circuit.append(cirq.CZ(q1, q2))
            elif g == "swap":
                circuit.append(cirq.SWAP(q1, q2))
    return circuit, qubits


# In[10]:


def add_noise(circuit: cirq.Circuit, qubits: cirq.LineQubit, noise_type: str, p: float) -> cirq.Circuit:
    noisy = cirq.Circuit()
    kinds = ["bitflip", "depolarizing", "amp_damp", "phase_damp"]

    for moment in circuit:
        noisy += moment
        if noise_type == "mixed":
            kind = random.choice(kinds)
        else:
            kind = noise_type

        if kind == "bitflip":
            noisy += cirq.bit_flip(p).on_each(*qubits)
        elif kind == "depolarizing":
            noisy += cirq.depolarize(p).on_each(*qubits)
        elif kind == "amp_damp":
            noisy += cirq.amplitude_damp(p).on_each(*qubits)
        elif kind == "phase_damp":
            noisy += cirq.phase_damp(p).on_each(*qubits)

    return noisy


# In[11]:


def simulate_density_matrix(circuit):
    simulator = cirq.DensityMatrixSimulator()
    result = simulator.simulate(circuit)
    return result.final_density_matrix


# In[12]:


def generate_dataset(n_samples=1000, n_qubits=3, noise_level=0.05):
    X, Y = [], []
    for _ in range(n_samples):
        circuit, qubits = random_circuit(n_qubits, depth=8)
        rho_clean = simulate_density_matrix(circuit)
        noisy_circuit = add_noise(circuit, qubits, "mixed", noise_level)
        rho_noisy = simulate_density_matrix(noisy_circuit)

        X.append(np.stack([rho_noisy.real, rho_noisy.imag], axis=-1))
        Y.append(np.stack([rho_clean.real, rho_clean.imag], axis=-1))
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


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

# In[13]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


# In[14]:


class QuantumDenoisingDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx].permute(2, 0, 1), self.Y[idx].permute(2, 0, 1)
        


# In[15]:


class PatchEmbed(nn.Module):
    def __init__(self, in_chans=2, embed_dim=64, patch_size=2):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                           
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1,2)            
        return x, (H, W)


# In[16]:


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


# In[17]:


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
        out = self.deproj(tokens)
        return out  # (B, 2, H_orig, W_orig)


# In[21]:


def fidelity_loss(pred, target):
    # pred,target: (B, 2, H, W)
    pred_c = pred[:,0] + 1j * pred[:,1]
    targ_c = target[:,0] + 1j * target[:,1]
    # Frobenius inner product
    num = torch.real(torch.sum(pred_c.conj() * targ_c, dim=[1,2]))
    denom = torch.sqrt(torch.sum(torch.abs(pred_c)**2, dim=[1,2]) * torch.sum(torch.abs(targ_c)**2, dim=[1,2]))
    fid = num / (denom + 1e-8)
    return 1 - torch.mean(fid)


# In[20]:


def total_loss(pred, target, alpha=0.5):
    return alpha * F.mse_loss(pred, target) + (1 - alpha) * fidelity_loss(pred, target)


# In[23]:


# Generate data
X, Y = generate_dataset(n_samples=500, n_qubits=2, noise_level=0.1)

# Dataset and DataLoader
train_ds = QuantumDenoisingDataset(X, Y)
train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)

# Model + optimizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = QuantumTransformerDenoiser().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)


# In[24]:


for epoch in range(20):
    model.train()
    total = 0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = total_loss(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
    print(f"Epoch {epoch:02d} | Loss: {total/len(train_dl):.4f}")


# # Testing

# In[26]:


import matplotlib.pyplot as plt


# In[ ]:


# Test quantum simulator
X, Y = generate_dataset(n_samples=1, n_qubits=2)
plt.subplot(1,2,1)
plt.imshow(X[0,:,:,0], cmap='RdBu'); plt.title("Noisy (Re)")
plt.subplot(1,2,2)
plt.imshow(Y[0,:,:,0], cmap='RdBu'); plt.title("Clean (Re)")
plt.show()


# In[31]:


# Test the model

model.eval()
xb, yb = next(iter(train_dl))
with torch.no_grad():
    pred = model(xb.to(device)).cpu()

plt.subplot(1,3,1)
plt.imshow(xb[0,0], cmap="RdBu"); plt.title("Noisy (Re)")
plt.subplot(1,3,2)
plt.imshow(pred[0,0], cmap="RdBu"); plt.title("Predicted (Re)")
plt.subplot(1,3,3)
plt.imshow(yb[0,0], cmap="RdBu"); plt.title("Clean (Re)")
plt.show()


# In[ ]:




