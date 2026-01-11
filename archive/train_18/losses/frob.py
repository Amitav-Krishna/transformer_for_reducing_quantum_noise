"""
Frobenius fidelity loss for quantum density matrices.

Computes cosine similarity between complex density matrices.
"""

import torch
import torch.nn as nn


class FrobeniusFidelityLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        # pred/target shape: (B, 2, 32, 32) where dim 1 is [real, imag]
        a = pred[:, 0] + 1j * pred[:, 1]
        b = target[:, 0] + 1j * target[:, 1]

        num = torch.real(torch.sum(a.conj() * b, dim=(-1, -2)))
        denom = torch.sqrt(
            torch.sum(torch.abs(a) ** 2, dim=(-1, -2))
            * torch.sum(torch.abs(b) ** 2, dim=(-1, -2))
            + self.eps
        )

        fid = torch.clamp(num / denom, -1, 1)
        return 1 - fid.mean()
