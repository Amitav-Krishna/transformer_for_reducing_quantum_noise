import torch.nn as nn
import torch

class FrobeniusFidelityLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        a = pred[..., 0] + 1j * pred[..., 1]
        b = target[..., 0] + 1j * target[..., 1]

        num = torch.real(torch.sum(a.conj() * b, dim=(-1, -2)))
        denom = torch.sqrt(torch.sum(torch.abs(a)**2, dim=(-1,-2)) *
                           torch.sum(torch.abs(b)**2, dim=(-1,-2)) + self.eps)

        fid = torch.clamp(num / denom, -1, 1)
        return 1 - fid.mean()

