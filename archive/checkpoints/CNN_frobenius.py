import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict

########################################
# CNN (Sigmoid removed)
########################################

class CNNAutoencoder(nn.Module):
    def __init__(self, loss_fn=None):
        super().__init__()
        self.loss_fn = loss_fn

        # ----- ENCODER -----
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # ----- BOTTLENECK -----
        self.bottleneck = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # ----- DECODER (NO SIGMOID) -----
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(192, 96, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(96, 48, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(48, 2, kernel_size=3, padding=1),
        )

    def forward(self, x):
        enc = self.encoder(x)
        z = self.bottleneck(enc)
        out = self.decoder(z)
        return out

    def compute_loss(self, prediction, target):
        if self.loss_fn is None:
            raise ValueError("No loss function provided.")
        return self.loss_fn(prediction, target)


########################################
# Uhlmann fidelity loss
########################################
class UhlmannFidelityLoss(nn.Module):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, rho_hat, rho_true):
        B, C, H, W = rho_hat.shape
        assert C == 2

        # Convert to complex64
        rho_hat_c = (rho_hat[:,0] + 1j * rho_hat[:,1]).to(torch.complex64)
        rho_true_c = (rho_true[:,0] + 1j * rho_true[:,1]).to(torch.complex64)

        fidelities = []

        for b in range(B):
            r = rho_hat_c[b]
            s = rho_true_c[b]

            # sqrt(rho_hat)
            eigvals_r, eigvecs_r = torch.linalg.eigh(
                r + self.eps * torch.eye(H, dtype=torch.complex64, device=r.device)
            )

            diag_r = torch.diag(eigvals_r.clamp(min=0).sqrt()).to(torch.complex64)
            sqrt_r = eigvecs_r @ diag_r @ eigvecs_r.conj().T

            # middle product
            mid = sqrt_r @ s @ sqrt_r

            eigvals_mid, eigvecs_mid = torch.linalg.eigh(
                mid + self.eps * torch.eye(H, dtype=torch.complex64, device=r.device)
            )

            diag_mid = torch.diag(eigvals_mid.clamp(min=0).sqrt()).to(torch.complex64)
            trace_sqrt = diag_mid.sum().real

            fidelities.append(trace_sqrt * trace_sqrt)

        fid = torch.stack(fidelities)
        return 1 - fid.mean()

########################################
# Frobenius norm
########################################

class FrobeniusFidelityLoss(nn.Module):
    """
    Loss = 1 - mean Frobenius fidelity over batch.

    pred, target: (B, 2, H, W) with channels [real, imag]
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        # Convert to complex: (B, H, W)
        a = pred[:, 0] + 1j * pred[:, 1]
        b = target[:, 0] + 1j * target[:, 1]

        # <A, B>_F
        num = torch.real(torch.sum(a.conj() * b, dim=(1, 2)))

        # ||A||_F * ||B||_F
        norm_a = torch.sqrt(torch.sum(torch.abs(a) ** 2, dim=(1, 2)) + self.eps)
        norm_b = torch.sqrt(torch.sum(torch.abs(b) ** 2, dim=(1, 2)) + self.eps)
        denom = norm_a * norm_b + self.eps

        fid = num / denom   # should be in [-1, 1], but clamp just in case
        fid = torch.clamp(fid, -1.0, 1.0)

        # We want to maximize fidelity → minimize 1 - fidelity
        return 1.0 - fid.mean()

########################################
# Dataset loader
########################################

class ChunkDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X.float()
        self.Y = Y.float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].permute(2,0,1)  # (2,H,W)
        y = self.Y[idx].permute(2,0,1)
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
        noise_type = parts[0]
        noise_level = parts[1]
        key = f"{noise_type}_{noise_level}"
        buckets[key].append(fname)

for key in buckets:
    buckets[key].sort()

balanced_files = []
for key, flist in buckets.items():
    balanced_files.extend(flist[:target_files_per_cell])

print("Selected balanced files:", len(balanced_files))


########################################
# Initialize model with Uhlmann loss
########################################

loss_fn = FrobeniusFidelityLoss()
model = CNNAutoencoder(loss_fn=loss_fn).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

EPOCHS = 100
BATCH = 16

# Make checkpoint directory
os.makedirs("checkpoints", exist_ok=True)


########################################
# CSV Logging
########################################

with open("training_loss_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "chunk_file", "batch_idx", "loss"])


########################################
# Training loop
########################################

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}")

    for fname in balanced_files:
        path = os.path.join(dataset_dir, fname)
        blob = torch.load(path)

        X = blob["X"]
        Y = blob["Y"]

        ds = ChunkDataset(X, Y)
        loader = DataLoader(ds, batch_size=BATCH, shuffle=True)

        for batch_idx, (x, y) in enumerate(loader):
            x = x.cuda()
            y = y.cuda()

            pred = model(x)
            loss = model.compute_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with open("training_loss_log.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, fname, batch_idx, loss.item()])

        print(f"  finished {fname}  final loss={loss.item():.6f}")

    # Save checkpoint
    ckpt_name = f"checkpoints/CNN_Uhlmann_{epoch+1}.pt"
    torch.save(model.state_dict(), ckpt_name)
    print(f"Saved checkpoint {ckpt_name}")

