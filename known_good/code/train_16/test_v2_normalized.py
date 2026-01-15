"""Test with normalized inputs - 800 train samples."""

import sys

sys.path.insert(0, "/workspace")
import torch
from torch.utils.data import DataLoader, TensorDataset
from train_16.models.axial_v2 import AxialTransformerV2
from losses.frob import FrobeniusFidelityLoss

device = torch.device("cuda")
loss_fn = FrobeniusFidelityLoss()

d = torch.load(
    "/workspace/dataset_8qubit_float64/cell00_chunk01.pt", weights_only=False
)
train_n = d["X"][:800].permute(0, 3, 1, 2).float().to(device)
train_c = d["Y"][:800].permute(0, 3, 1, 2).float().to(device)
val_n = d["X"][800:].permute(0, 3, 1, 2).float().to(device)
val_c = d["Y"][800:].permute(0, 3, 1, 2).float().to(device)


def normalize_to_unit(x):
    norms = x.flatten(1).norm(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
    return x / (norms + 1e-8)


train_n_norm = normalize_to_unit(train_n)
train_c_norm = normalize_to_unit(train_c)
val_n_norm = normalize_to_unit(val_n)
val_c_norm = normalize_to_unit(val_c)

print(f"Train: {len(train_n)}, Val: {len(val_n)}")
print(f"Baseline: {1 - loss_fn(val_n_norm, val_c_norm):.4f}")
print(flush=True)

model = AxialTransformerV2(
    embed_dim=256, ffn_dim=1024, num_heads=8, num_layers=4, dropout=0.1
).to(device)
loader = DataLoader(
    TensorDataset(train_n_norm, train_c_norm), batch_size=8, shuffle=True
)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

for ep in range(1, 101):
    model.train()
    for xb, yb in loader:
        opt.zero_grad()
        loss_fn(model(xb), yb).backward()
        opt.step()

    if ep % 10 == 0:
        model.eval()
        with torch.no_grad():
            train_fid = 1 - loss_fn(model(train_n_norm[:200]), train_c_norm[:200])
            val_fid = 1 - loss_fn(model(val_n_norm), val_c_norm)
        print(f"Epoch {ep}: train={train_fid:.4f}, val={val_fid:.4f}", flush=True)
