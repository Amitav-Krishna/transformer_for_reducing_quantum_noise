import os
import torch
from torch.utils.data import DataLoader

from training_loop.dataset.ChunkDataset import ChunkDataset
from training_loop.dataset.csv_logger import CSVLogger
from training_loop.evaluate_on_chunks import evaluate_on_chunks


def train_single_experiment(name, config, train_chunks, val_chunks, device):

    arch = config["arch"]
    loss = config["loss"]

    # Directories
    ckpt_dir = f"checkpoints_2/{name}"
    os.makedirs(ckpt_dir, exist_ok=True)

    csv_logger = CSVLogger(csv_dir="csvs_2", name=name)

    # Model + Optimizer
    model = config["create_model"]().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    EPOCHS = 100
    BATCH = 32 if arch == "cnn" else 8

    print(f"\n==============================")
    print(f"Training {name}")
    print(f"Architecture: {arch} | Loss: {loss}")
    print(f"Batch size: {BATCH}")
    print(f"==============================\n")

    best_val = float("inf")

    for epoch in range(1, EPOCHS+1):
        print(f"Epoch {epoch}")

        # Training loop
        for c_idx, (X, Y) in enumerate(train_chunks):
            ds = ChunkDataset(X, Y, arch)
            loader = DataLoader(ds, batch_size=BATCH, shuffle=True)

            for b_idx, (x, y) in enumerate(loader):
                x, y = x.to(device), y.to(device)

                pred = model(x)
                train_loss = model.compute_loss(pred, y)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                # CSV logging (SAFE)
                csv_logger.log_train(epoch, c_idx, b_idx, float(train_loss.item()))

            print(f"  chunk {c_idx} final batch loss = {train_loss.item():.6f}")

        # Validation
        val_loss = evaluate_on_chunks(model, val_chunks, arch, device, BATCH)
        print(f"Validation loss: {val_loss:.6f}")

        csv_logger.log_val(epoch, float(val_loss))

        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"{ckpt_dir}/best.pt")
            print(f"{epoch} is the best checkpoint so far.")
        torch.save(model.state_dict(), f"{ckpt_dir}/{epoch}.pt")
