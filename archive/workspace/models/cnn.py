import torch
import torch.nn as nn

class CNNAutoencoder(nn.Module):
    def __init__(self, loss_fn=None):
        super().__init__()
        self.loss_fn = loss_fn

        self.encoder = nn.Sequential(
            nn.Conv2d(2, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),    # 2 -> 48

            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),    # 48 → 96

            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),    # 96 -> 192
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),  # 192 -> 96
            nn.ConvTranspose2d(192, 96, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode="nearest"),  # 96 -> 48
            nn.ConvTranspose2d(96, 48, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode="nearest"),  # 48 -> 2
            nn.ConvTranspose2d(48, 2, kernel_size=3, padding=1),
        )

    def forward(self, x):
        z = self.bottleneck(self.encoder(x))
        return self.decoder(z)

    def compute_loss(self, pred, target):
        return self.loss_fn(pred, target)

