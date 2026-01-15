import torch

@torch.no_grad()
def frob_fidelity_batch(rho_pred, rho_true, eps=1e-12):
    """
    Computes Frobenius fidelity proxy between predicted and true density matrices.
    rho_* are (B, 2, 32, 32) with [real, imag] channels.
    Returns shape (B,) tensor of fidelities.
    """

    # Convert back to complex matrices
    a = rho_pred[:, 0] + 1j * rho_pred[:, 1]
    b = rho_true[:, 0] + 1j * rho_true[:, 1]

    # Frobenius inner product
    num = torch.real(torch.sum(a.conj() * b, dim=(1, 2)))

    # Norms
    denom = torch.sqrt(torch.sum(torch.abs(a)**2, dim=(1, 2)) *
                       torch.sum(torch.abs(b)**2, dim=(1, 2)))

    return (num / (denom + eps)).clamp(-1, 1)  # range [-1,1]


@torch.no_grad()
def evaluate_fidelity_on_chunks(model, chunks, arch, device, batch):
    """
    Computes MEAN fidelity over a list of chunks.
    Returns mean_fidelity, std_fidelity
    """
    from training_loop.dataset.ChunkDataset import ChunkDataset
    from torch.utils.data import DataLoader

    model.eval()
    all_fid = []

    for X, Y in chunks:
        ds = ChunkDataset(X, Y, arch)
        loader = DataLoader(ds, batch_size=batch, shuffle=False)

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)      # (B,2,32,32)

            fid = frob_fidelity_batch(pred, y)  # (B,)
            all_fid.append(fid.cpu())

    all_fid = torch.cat(all_fid)

    model.train()
    return all_fid.mean().item(), all_fid.std().item()
