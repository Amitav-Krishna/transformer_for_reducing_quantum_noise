import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict

########################################
# Transformer Autoencoder (your version)
# Output Sigmoid removed
########################################

class TransformerAutoencoder(nn.Module):
    def __init__(self, loss_fn=None):
        super().__init__()
        self.loss_fn = loss_fn

        # ----- SPEC -----
        self.seq_length = 1024  # 32x32 flattened
        self.input_dim = 2      # real + imag
        self.embed_dim = 32
        self.ffn_dim = 64
        self.num_heads = 4
        self.num_layers = 4

        # Project each token (2 values) into 32-dim embedding
        self.input_proj = nn.Linear(self.input_dim, self.embed_dim)

        # Positional encoding
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.seq_length, self.embed_dim)
        )

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ffn_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        # Bottleneck 32 -> 16 -> 32
        self.bottleneck_down = nn.Linear(self.embed_dim, 16)
        self.bottleneck_up = nn.Linear(16, self.embed_dim)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ffn_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=self.num_layers
        )

        # Output: 32 → 2 (real+imag)
        self.output_proj = nn.Linear(self.embed_dim, self.input_dim)
        # self.output_activation = nn.Sigmoid()   # Removed

    def forward(self, x):
        # x shape: (B, 1024, 2)
        x = self.input_proj(x) + self.pos_embedding
        enc = self.encoder(x)
        z = self.bottleneck_up(self.bottleneck_down(enc))
        dec = self.decoder(z, enc)
        out = self.output_proj(dec)
        return out

    def compute_loss(self, prediction, target):
        if self.loss_fn is None:
            raise ValueError("No loss function provided.")
        return self.loss_fn(prediction, target)


########################################
# Frobenius Fidelity Loss
########################################

class FrobeniusFidelityLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        # pred, target: (B, 1024, 2)
        a = pred[...,0] + 1j * pred[...,1]
        b = target[...,0] + 1j * target[...,1]

        num = torch.real(torch.sum(a.conj() * b, dim=1))
        norm_a = torch.sqrt(torch.sum(torch.abs(a)**2, dim=1) + self.eps)
        norm_b = torch.sqrt(torch.sum(torch.abs(b)**2, dim=1) + self.eps)

        fid = num / (norm_a * norm_b + self.eps)
        fid = fid.clamp(-1, 1)
        return 1 - fid.mean()


########################################
# Dataset loader
########################################

class ChunkDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        # X, Y: (N, 32, 32, 2)
        self.X = X.float()
        self.Y = Y.float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].reshape(1024, 2)  # flatten
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
        key = f"{parts[0]}_{parts[1]}"
        buckets[key].append(fname)

for key in buckets:
    buckets[key].sort()

balanced_files = []
for k, v in buckets.items():
    balanced_files.extend(v[:target_files_per_cell])

print("Selected balanced files:", len(balanced_files))


########################################
# Init Model + Optimizer
########################################

loss_fn = FrobeniusFidelityLoss()
model = TransformerAutoencoder(loss_fn=loss_fn).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

EPOCHS = 100
BATCH = 16

os.makedirs("checkpoints", exist_ok=True)


########################################
# CSV Logging
########################################

with open("training_loss_log_transformer_frobenius.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "chunk_file", "batch_idx", "loss"])


########################################
# Training Loop
########################################

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}")

    for fname in balanced_files:
        blob = torch.load(os.path.join(dataset_dir, fname))
        ds = ChunkDataset(blob["X"], blob["Y"])
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
                csv.writer(f).writerow([epoch+1, fname, batch_idx, loss.item()])

        print(f"  finished {fname}   final loss={loss.item():.6f}")

    ckpt = f"checkpoints/Transformer_Frob_{epoch+1}.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"Saved {ckpt}")

