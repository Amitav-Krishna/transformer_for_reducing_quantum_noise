"""
Loss functions for Pauli basis representation.

For real-valued Pauli coefficients, the Frobenius inner product in density matrix
space corresponds to the Euclidean inner product in Pauli space (with normalization).
"""

import torch
import torch.nn as nn


class PauliFrobeniusLoss(nn.Module):
    """
    Frobenius-like loss for Pauli basis representation.

    Computes cosine similarity between predicted and target Pauli coefficients,
    which is equivalent to the normalized Frobenius inner product.

    Input shape: (B, 1024) real-valued Pauli coefficients
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        # pred, target shape: (B, 1024)

        # Cosine similarity (equivalent to normalized Frobenius inner product)
        num = torch.sum(pred * target, dim=-1)
        denom = torch.sqrt(
            torch.sum(pred ** 2, dim=-1) * torch.sum(target ** 2, dim=-1) + self.eps
        )

        similarity = num / denom
        similarity = torch.clamp(similarity, -1, 1)

        # Loss = 1 - similarity (so minimizing loss maximizes similarity)
        return (1 - similarity).mean()


class PauliMSELoss(nn.Module):
    """
    Simple MSE loss for Pauli coefficients.

    Input shape: (B, 1024)
    """

    def forward(self, pred, target):
        return nn.functional.mse_loss(pred, target)
