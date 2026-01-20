import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict

########################################
# Transformer Autoencoder (your version)
# Output Sigmoid REMOVED (physically meaningless)
########################################

class TransformerAutoencoder(nn.Module):
    def __init__(self, loss_fn=None):
        super().__init__()
        self.loss_fn = loss_fn

        self.seq_length = 1024  # 32*32 flattened
        self.input_dim = 2      # real + imag
        self.embed_dim = 32
        self.ffn_dim = 64
        self.num_heads = 4
        self.num_layers = 4

        self.input_proj = nn.Linear(self.input_dim, self.embed_dim)

        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.seq_length, self.embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ffn_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.bottleneck_down = nn.Linear(self.embed_dim, 16)
        self.bottleneck_up   = nn.Linear(16, self.embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ffn_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)

        self.output_proj = nn.Linear(self.embed_dim, self.input_dim)

    def forward(self, x):
        # x: (B, 1024, 2)
        x = self.input_proj(x) + self.pos_embedding
        enc = self.encoder(x)
        z = self.bottleneck_up(self.bottleneck_down(enc))
        dec = self.decoder(z, enc)
        return self.output_proj(dec)

    def compute_loss(self, prediction, target):
        return self.loss_fn(prediction, target)


########################################
# Frobenius Fidelity Loss
########################################

class FrobeniusFidelityLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        # (B, 1024, 2)
        a = pred[...,0] + 1j * pred[...,1]
        b = target[...,0] + 1j * target[...,1]

        num = torch.real(torch.sum(a.conj() * b, dim=1))
        norm_a = torch.sqrt(torch.sum(torch.abs(a)**2, dim=1) + self.eps)
        norm_b = torch.sqrt(torch.sum(torch.abs(b)**2, dim=1) + self.eps)

        fid = num / (norm_a * norm_b + self.eps)
        fid = fid.clamp(-1, 1)

        return 1 - fid.mean()


########################################
# Physics-Informed Density Matrix Loss
########################################

class DensityMatrixPhysicsLoss(nn.Module):
    def __init__(self, λ_trace=1.0, λ_herm=1.0, λ_psd=1.0):
        super().__init__()
        self.λ_trace = λ_trace
        self.λ_herm = λ_herm
        self.λ_psd = λ_psd

    def forward(self, rho_flat):
        # rho_flat: (B, 1024, 2)
        B = rho_flat.shape[0]
        rho = (rho_flat[...,0] + 1j * rho_flat[...,1]).reshape(B, 32, 32)

        # Hermiticity
        herm_diff = rho - rho.conj().transpose(-1, -2)
        herm_penalty = torch.mean(torch.abs(herm_diff)**2)

        # Trace
        tr = rho.diagonal(dim1=-2, dim2=-1).sum(-1).real
        trace_penalty = torch.mean((tr - 1.0)**2)

        # PSD: penalize negative eigenvalues
        rho_herm = 0.5 * (rho + rho.conj().transpose(-1, -2))
        eigvals = torch.linalg.eigvalsh(rho_herm)
        psd_penalty = torch.mean(torch.relu(-eigvals)**2)

        return (
            self.λ_herm * herm_penalty +
            self.λ_trace * trace_penalty +
            self.λ_psd * psd_penalty
        )


########################################
# Combined Physics-Informed Frobenius Loss
########################################

class Totalphysics_informedLoss(nn.Module):
    def __init__(
        self,
        fidelity_weight=1.0,
        physics_weight=0.1,
        eps=1e-8,
        λ_trace=1.0,
        λ_herm=1.0,
        λ_psd=1.0,
    ):
        super().__init__()
        self.fidelity = FrobeniusFidelityLoss(eps)
        self.physics = DensityMatrixPhysicsLoss(
            λ_trace=λ_trace,
            λ_herm=λ_herm,
            λ_psd=λ_psd
        )
        self.w_fid = fidelity_weight
        self.w_phys = physics_weight

    def forward(self, pred, target):
        L_fid = self.fidelity(pred, target)
        L_phys = self.physics(pred)
        return self.w_fid * L_fid + self.w_phys * L_phys


########################################
# Dataset Loader
########################################

class ChunkDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X.float()
        self.Y = Y.float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].reshape(1024, 2)
        y = self.Y[idx].reshape(1024, 2)
        return x, y


########################################
# Load balanced dataset files
########################################

dataset_dir = "dataset_smaller"
target_files_per_cell = 50

files = os.listdir(dataset_dir)
buckets = defaultdict(list)

for fname in files:
    if fname.endswith(".pt"):
        parts = fname.split("_")
        buckets[f"{parts[0]}_{parts[1]}"].append(fname)

for k in buckets:
    buckets[k].sort()

balanced_files = []
for k, flist in buckets.items():
    balanced_files.extend(flist[:target_files_per_cell])

print("Selected balanced files:", len(balanced_files))


########################################
# Init model + optimizer
########################################

loss_fn = Totalphysics_informedLoss(
    fidelity_weight=1.0,
    physics_weight=0.1,
)

model = TransformerAutoencoder(loss_fn=loss_fn).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

EPOCHS = 100
BATCH = 8   # transformers eat VRAM, reduce batch size

os.makedirs("checkpoints", exist_ok=True)


########################################
# Logging
########################################

with open("training_loss_log_transformer_physics_informed.csv", "w", newline="") as f:
    csv.writer(f).writerow(["epoch", "chunk", "batch", "loss"])


########################################
# Training Loop
########################################

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}")

    for fname in balanced_files:
        blob = torch.load(os.path.join(dataset_dir, fname))
        loader = DataLoader(ChunkDataset(blob["X"], blob["Y"]), batch_size=BATCH, shuffle=True)

        for batch_idx, (x, y) in enumerate(loader):
            x = x.cuda()
            y = y.cuda()

            pred = model(x)
            loss = model.compute_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with open("training_loss_log_transformer_physics_informed.csv", "a", newline="") as f:
                csv.writer(f).writerow([epoch+1, fname, batch_idx, loss.item()])

        print(f"  finished {fname}, final loss={loss.item():.6f}")

    ckpt = f"checkpoints/Transformer_physics_informed_{epoch+1}.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"Saved {ckpt}")

