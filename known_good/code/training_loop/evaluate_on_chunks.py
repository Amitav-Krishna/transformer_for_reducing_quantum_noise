import torch
from torch.utils.data import DataLoader
from training_loop.dataset.ChunkDataset import ChunkDataset   


def evaluate_on_chunks(model, chunks, arch, device, batch):
    model.eval()
    total = 0
    count = 0
    with torch.no_grad():
        for X, Y in chunks:
            ds = ChunkDataset(X, Y, arch)
            loader = DataLoader(ds, batch_size=batch, shuffle=False)
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)

                pred = model(x)
                loss = model.compute_loss(pred, y)

                total += loss.item()
                count += 1

    model.train()
    return total / count
