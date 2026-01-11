"""
MLP Autoencoder for density matrix denoising.

Unconstrained output - use post-hoc projection for physical validity.
"""

import torch
import torch.nn as nn


class MLPAutoencoder(nn.Module):
    """
    MLP autoencoder with unconstrained output.

    Architecture:
        2048 -> 512 -> 256 -> 512 -> 2048 -> (2, 32, 32)

    Output requires post-hoc projection to ensure valid density matrix.
    """

    def __init__(self, loss_fn=None, dropout=0.1):
        super().__init__()
        self.loss_fn = loss_fn

        self.input_dim = 32 * 32 * 2  # 2048
        self.output_dim = 32 * 32 * 2  # 2048

        # Encoder-decoder network
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, self.output_dim),
        )

    def forward(self, x):
        B = x.shape[0]
        # Flatten: (B, 2, 32, 32) -> (B, 2048)
        x = x.reshape(B, -1)

        # Network produces output
        out = self.network(x)

        # Reshape to (B, 2, 32, 32)
        out = out.reshape(B, 2, 32, 32)

        return out

    def compute_loss(self, pred, target):
        return self.loss_fn(pred, target)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/work/codage/transformer_qnr')
    from losses.frob import FrobeniusFidelityLoss

    model = MLPAutoencoder(loss_fn=FrobeniusFidelityLoss())
    print(f"MLPAutoencoder parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(4, 2, 32, 32)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
