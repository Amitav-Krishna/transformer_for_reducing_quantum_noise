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

print(f"Baseline val: {1 - loss_fn(val_n, val_c):.4f}")

model = AxialTransformerV2(
    embed_dim=256, ffn_dim=1024, num_heads=8, num_layers=4, dropout=0.1, loss_fn=loss_fn
).to(device)
loader = DataLoader(TensorDataset(train_n, train_c), batch_size=8, shuffle=True)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

for ep in range(1, 51):
    model.train()
    for xb, yb in loader:
        opt.zero_grad()
        loss_fn(model(xb), yb).backward()
        opt.step()
    if ep % 10 == 0:
        model.eval()
        with torch.no_grad():
            va = 1 - loss_fn(model(val_n), val_c)
        print(f"Epoch {ep}: val={va:.4f}")
