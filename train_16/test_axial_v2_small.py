"""
Test Axial V2 on small subset to check generalization.
Same setup as before: 100 train, 100 val from same chunk.
"""

import sys

sys.path.insert(0, "/workspace")

import torch
from torch.utils.data import DataLoader, TensorDataset

from train_16.models.axial_v2 import AxialTransformerV2
from losses.frob import FrobeniusFidelityLoss


device = torch.device("cuda")
loss_fn = FrobeniusFidelityLoss()

# Load data - same split as before
d = torch.load(
    "/workspace/dataset_8qubit_float64/cell00_chunk01.pt", weights_only=False
)
train_n = d["X"][:100].permute(0, 3, 1, 2).float().to(device)
train_c = d["Y"][:100].permute(0, 3, 1, 2).float().to(device)
val_n = d["X"][100:200].permute(0, 3, 1, 2).float().to(device)
val_c = d["Y"][100:200].permute(0, 3, 1, 2).float().to(device)

print(f"Train: {len(train_n)}, Val: {len(val_n)}")
print(f"Baseline train fidelity: {1 - loss_fn(train_n, train_c):.4f}")
print(f"Baseline val fidelity: {1 - loss_fn(val_n, val_c):.4f}")
print()

# Model with new settings
model = AxialTransformerV2(
    embed_dim=256,
    ffn_dim=1024,
    num_heads=8,
    num_layers=4,
    dropout=0.1,
    loss_fn=loss_fn,
).to(device)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print()

loader = DataLoader(TensorDataset(train_n, train_c), batch_size=4, shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

print(
    "Training with: residual prediction, concat+linear merge, dropout=0.1, weight_decay=0.01"
)
print()

for ep in range(1, 101):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()

    if ep % 20 == 0:
        model.eval()
        with torch.no_grad():
            train_fid = 1 - loss_fn(model(train_n), train_c)
            val_fid = 1 - loss_fn(model(val_n), val_c)
        print(f"Epoch {ep}: train={train_fid:.4f}, val={val_fid:.4f}")
