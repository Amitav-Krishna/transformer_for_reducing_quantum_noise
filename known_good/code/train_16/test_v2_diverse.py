"""Test with diverse noise types from multiple cells."""

import sys

sys.path.insert(0, "/workspace")
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from train_16.models.axial_v2 import AxialTransformerV2
from losses.frob import FrobeniusFidelityLoss

device = torch.device("cuda")
loss_fn = FrobeniusFidelityLoss()

# Load from multiple cells (different noise types/levels)
dataset_path = Path("/workspace/dataset_8qubit_float64")
train_chunks = [
    "cell00_chunk01.pt",  # depolarizing 0.05
    "cell01_chunk01.pt",  # depolarizing 0.1
    "cell10_chunk01.pt",  # phase_damping 0.15
    "cell20_chunk01.pt",  # likely different
]

all_X, all_Y = [], []
for name in train_chunks:
    p = dataset_path / name
    if p.exists():
        d = torch.load(p, weights_only=False)
        all_X.append(d["X"][:200])  # 200 from each = 800 total
        all_Y.append(d["Y"][:200])
        meta = d.get("meta", [{}])[0]
        print(f"{name}: {meta}")

train_X = torch.cat(all_X).permute(0, 3, 1, 2).float().to(device)
train_Y = torch.cat(all_Y).permute(0, 3, 1, 2).float().to(device)

# Val from different chunks of same cells
val_chunks = [
    "cell00_chunk02.pt",
    "cell01_chunk02.pt",
    "cell10_chunk02.pt",
    "cell20_chunk02.pt",
]

val_X, val_Y = [], []
for name in val_chunks:
    p = dataset_path / name
    if p.exists():
        d = torch.load(p, weights_only=False)
        val_X.append(d["X"][:50])  # 50 from each = 200 total
        val_Y.append(d["Y"][:50])

val_X = torch.cat(val_X).permute(0, 3, 1, 2).float().to(device)
val_Y = torch.cat(val_Y).permute(0, 3, 1, 2).float().to(device)

print(f"\nTrain: {len(train_X)}, Val: {len(val_X)}")
print(f"Baseline val: {1 - loss_fn(val_X, val_Y):.4f}")

model = AxialTransformerV2(
    embed_dim=256, ffn_dim=1024, num_heads=8, num_layers=4, dropout=0.1, loss_fn=loss_fn
).to(device)
loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=8, shuffle=True)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

print()
for ep in range(1, 51):
    model.train()
    for xb, yb in loader:
        opt.zero_grad()
        loss_fn(model(xb), yb).backward()
        opt.step()
    if ep % 10 == 0:
        model.eval()
        with torch.no_grad():
            tr = 1 - loss_fn(model(train_X[:200]), train_Y[:200])
            va = 1 - loss_fn(model(val_X), val_Y)
        print(f"Epoch {ep}: train={tr:.4f}, val={va:.4f}")
