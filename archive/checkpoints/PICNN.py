import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict

########################################
# CNN Autoencoder (Sigmoid removed)
########################################

class CNNAutoencoder(nn.Module):
    def __init__(self, loss_fn=None):
        super().__init__()
        self.loss_fn = loss_fn

        # ----- ENCODER -----
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),          # 2 x 32 x 32 -> 48 x 16 x 16

            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),          # 96 x 8 x 8

            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),          # 192 x 4 x 4
        )

        # ----- BOTTLENECK -----
        self.bottleneck = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # ----- DECODER (NO SIGMOID) -----
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),   # 4 -> 8
            nn.ConvTranspose2d(192, 96, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),   # 8 -> 16
            nn.ConvTranspose2d(96, 48, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),   # 16 -> 32
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
# Uhlmann fidelity (KEPT for reference, NOT used)
########################################

class UhlmannFidelityLoss(nn.Module):
    """
    Full Uhlmann fidelity. This is numerically and autograd-wise
    fragile because it depends on eigenvectors' arbitrary phases.
    We are NOT using this in training.
    """
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, rho_hat, rho_true):
        # rho_hat, rho_true: (B, 2, N, N) -> complex (B, N, N)
        B, C, H, W = rho_hat.shape
        assert C == 2

        rho_hat_c = (rho_hat[:, 0] + 1j * rho_hat[:, 1]).to(torch.complex64)
        rho_true_c = (rho_true[:, 0] + 1j * rho_true[:, 1]).to(torch.complex64)

        fidelities = []
        eye = torch.eye(H, dtype=torch.complex64, device=rho_hat.device)

        for b in range(B):
            r = rho_hat_c[b]
            s = rho_true_c[b]

            # sqrt(rho_hat)
            eigvals_r, eigvecs_r = torch.linalg.eigh(r + self.eps * eye)
            diag_r = torch.diag(eigvals_r.clamp(min=0).sqrt())
            sqrt_r = eigvecs_r @ diag_r @ eigvecs_r.conj().T

            # mid = sqrt(r) * s * sqrt(r)
            mid = sqrt_r @ s @ sqrt_r

            eigvals_mid, eigvecs_mid = torch.linalg.eigh(mid + self.eps * eye)
            diag_mid = torch.diag(eigvals_mid.clamp(min=0).sqrt())
            trace_sqrt = diag_mid.sum().real

            fidelities.append(trace_sqrt * trace_sqrt)

        fid = torch.stack(fidelities)
        return 1 - fid.mean()


########################################
# Frobenius fidelity
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

        fid = num / denom   # cosine similarity in Frobenius space
        fid = torch.clamp(fid, -1.0, 1.0)

        # We want to maximize fidelity → minimize 1 - fidelity
        return 1.0 - fid.mean()


########################################
# Physics-informed penalties
########################################

class DensityMatrixPhysicsLoss(nn.Module):
    """
    Penalize violations of:
      a) Hermiticity
      b) Unit trace
      c) Positive semidefiniteness (negative eigenvalues)

    Input: rho_2ch with shape (B, 2, N, N) [real, imag]
    """
    def __init__(self, λ_trace=1.0, λ_herm=1.0, λ_psd=1.0):
        super().__init__()
        self.λ_trace = λ_trace
        self.λ_herm = λ_herm
        self.λ_psd = λ_psd

    def forward(self, rho_2ch):
        # Convert to complex: (B, N, N)
        rho = rho_2ch[:, 0] + 1j * rho_2ch[:, 1]

        # Hermiticity penalty: ||ρ - ρ†||_F^2
        herm_diff = rho - rho.conj().transpose(-1, -2)
        herm_penalty = torch.mean(torch.abs(herm_diff) ** 2)

        # Trace = 1 penalty: (Tr(ρ) - 1)^2
        trace = torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1).real
        trace_penalty = torch.mean((trace - 1.0) ** 2)

        # PSD penalty: sum of squared negative eigenvalues
        # Make it explicitly Hermitian for eigvalsh stability
        rho_herm = 0.5 * (rho + rho.conj().transpose(-1, -2))
        eigvals = torch.linalg.eigvalsh(rho_herm)
        psd_penalty = torch.mean(torch.relu(-eigvals) ** 2)

        return (
            self.λ_herm * herm_penalty
            + self.λ_trace * trace_penalty
            + self.λ_psd * psd_penalty
        )


class TotalPhysicsReconstructionLoss(nn.Module):
    """
    Combined loss:
      L = w_fid * (1 - Frobenius fidelity) + w_phys * physics_penalty
    """
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
        self.fidelity_weight = fidelity_weight
        self.physics_weight = physics_weight

        self.recon_loss = FrobeniusFidelityLoss(eps=eps)
        self.physics_loss = DensityMatrixPhysicsLoss(
            λ_trace=λ_trace,
            λ_herm=λ_herm,
            λ_psd=λ_psd,
        )

    def forward(self, rho_hat_2ch, rho_true_2ch):
        L_recon = self.recon_loss(rho_hat_2ch, rho_true_2ch)
        L_phys = self.physics_loss(rho_hat_2ch)
        return self.fidelity_weight * L_recon + self.physics_weight * L_phys


########################################
# Dataset loader
########################################

class ChunkDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        # X, Y are already tensors in your .pt files
        self.X = X.float()
        self.Y = Y.float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # (H, W, 2) -> (2, H, W)
        x = self.X[idx].permute(2, 0, 1)
        y = self.Y[idx].permute(2, 0, 1)
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
# Initialize model with combined loss
########################################

loss_fn = TotalPhysicsReconstructionLoss(
    fidelity_weight=1.0,
    physics_weight=0.1,  # bump up or down later if needed
    eps=1e-8,
    λ_trace=1.0,
    λ_herm=1.0,
    λ_psd=1.0,
)

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

            with open("training_loss_log_PICNN.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, fname, batch_idx, loss.item()])

        print(f"  finished {fname}  final loss={loss.item():.6f}")

    # Save checkpoint
    ckpt_name = f"checkpoints/PICNN_{epoch+1}.pt"
    torch.save(model.state_dict(), ckpt_name)
    print(f"Saved checkpoint {ckpt_name}")

