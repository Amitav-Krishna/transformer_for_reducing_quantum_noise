from losses.physics import DensityMatrixPhysicsLoss
from losses.frob import FrobeniusFidelityLoss
import torch.nn as nn


class CompositePhysicsTotalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.fid = FrobeniusFidelityLoss()
        self.phys = DensityMatrixPhysicsLoss()
    def forward(self, pred, target):
        return self.fid(pred, target) + 0.1 * self.phys(pred)

