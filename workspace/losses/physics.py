import torch.nn as nn
import torch

class DensityMatrixPhysicsLoss(nn.Module):
    def __init__(self, λ_trace=1.0, λ_herm=1.0, λ_psd=1.0):
        super().__init__()
        self.λ_trace = λ_trace
        self.λ_herm = λ_herm
        self.λ_psd = λ_psd

    def forward(self, rho_2ch):
        real = rho_2ch[:, 0, :, :]
        imag = rho_2ch[:, 1, :, :]
        rho = (real + 1j * imag)

        herm = rho - rho.conj().transpose(-1,-2)
        herm_pen = torch.mean(torch.abs(herm)**2)

        tr = rho.diagonal(dim1=-2, dim2=-1).sum(-1).real
        trace_pen = torch.mean((tr - 1)**2)

        rho_sym = (rho + rho.conj().transpose(-1,-2)) / 2
        eigvals = torch.linalg.eigvalsh(rho_sym)
        psd_pen = torch.mean(torch.relu(-eigvals)**2)

        return (self.λ_herm * herm_pen +
                self.λ_trace * trace_pen +
                self.λ_psd * psd_pen)

