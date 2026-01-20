"""
Train the properly-initialized capacity-matched transformer (v4).

V4 fixes ALL the issues with v1-v3:
1. Depth-scaled initialization (1/sqrt(2*num_layers))
2. FFN ratio = 2x (prevents gradient starvation)
3. Pre-LN architecture for stability
4. Learned positional embeddings
5. HIGHER learning rate (1e-3 instead of 3e-4)
6. Linear warmup schedule (5 epochs)
7. Gradient clipping (max_norm=1.0)

Run:
    python train_transformer_matched_v4.py

Checkpoints: checkpoints_2/transformer_matched_v4_{frob,physics}/
Logs: csvs_2/transformer_matched_v4_{frob,physics}.csv
"""

import os
import torch
from torch.utils.data import DataLoader

from models.transformer_matched_v4 import TransformerAutoencoderMatchedV4
from losses.frob import FrobeniusFidelityLoss
from losses.total_physics_loss import CompositePhysicsTotalLoss
from training_loop.dataset.load_chunks import load_chunks
from training_loop.dataset.split_chunks import split_chunks
from training_loop.dataset.ChunkDataset import ChunkDataset
from training_loop.dataset.csv_logger import CSVLogger
from training_loop.evaluate_on_chunks import evaluate_on_chunks


def get_warmup_lr(epoch, batch_idx, total_batches, base_lr, warmup_epochs):
    """Linear warmup learning rate schedule."""
    if epoch <= warmup_epochs:
        # Linear warmup
        progress = (epoch - 1 + batch_idx / total_batches) / warmup_epochs
        return base_lr * max(progress, 0.01)  # Minimum 1% of base_lr
    else:
        # Constant after warmup
        return base_lr


def train_v4_experiment(name, config, train_chunks, val_chunks, device):
    """
    Training loop with warmup and gradient clipping for V4.
    """
    arch = config["arch"]
    loss_type = config["loss"]

    # Directories
    ckpt_dir = f"checkpoints_2/{name}"
    os.makedirs(ckpt_dir, exist_ok=True)

    # CSV logger
    csv_logger = CSVLogger(csv_dir="csvs_2", name=name)

    # Model
    model = config["create_model"]().to(device)

    # Training hyperparameters - KEY CHANGES FROM V3
    BASE_LR = config.get("lr", 1e-3)  # HIGHER than v3's 1e-4
    WARMUP_EPOCHS = 5
    MAX_GRAD_NORM = 1.0
    EPOCHS = 100
    BATCH = 8  # Transformer batch size

    # Optimizer (no scheduler - we do manual warmup)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BASE_LR,
        weight_decay=0.01,  # Light regularization
        betas=(0.9, 0.98),  # Standard transformer betas
        eps=1e-8
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"Training {name}")
    print(f"Architecture: {arch} | Loss: {loss_type}")
    print(f"Parameters: {num_params:,}")
    print(f"Batch size: {BATCH} | Base LR: {BASE_LR}")
    print(f"Warmup epochs: {WARMUP_EPOCHS} | Grad clip: {MAX_GRAD_NORM}")
    print(f"{'='*60}\n")

    best_val = float("inf")

    # Count total batches for warmup calculation
    total_train_samples = sum(X.shape[0] for X, Y in train_chunks)
    total_batches = total_train_samples // BATCH

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        print(f"Epoch {epoch}")

        for c_idx, (X, Y) in enumerate(train_chunks):
            ds = ChunkDataset(X, Y, arch)
            loader = DataLoader(ds, batch_size=BATCH, shuffle=True)

            for b_idx, (x, y) in enumerate(loader):
                x, y = x.to(device), y.to(device)

                # Warmup LR scheduling
                current_lr = get_warmup_lr(
                    epoch, batch_count, total_batches,
                    BASE_LR, WARMUP_EPOCHS
                )
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

                # Forward
                pred = model(x)
                train_loss = model.compute_loss(pred, y)

                # Backward with gradient clipping
                optimizer.zero_grad()
                train_loss.backward()

                # Gradient clipping - CRITICAL for stable training
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), MAX_GRAD_NORM
                )

                optimizer.step()

                epoch_loss += train_loss.item()
                batch_count += 1

                # Log
                csv_logger.log_train(epoch, c_idx, b_idx, float(train_loss.item()))

            print(f"  chunk {c_idx} final batch loss = {train_loss.item():.6f} "
                  f"(lr={current_lr:.2e}, grad_norm={grad_norm:.4f})")

        avg_train_loss = epoch_loss / batch_count
        print(f"Average training loss: {avg_train_loss:.6f}")

        # Validation
        model.eval()
        val_loss = evaluate_on_chunks(model, val_chunks, arch, device, BATCH)
        print(f"Validation loss: {val_loss:.6f}")

        csv_logger.log_val(epoch, float(val_loss))

        # Save best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(ckpt_dir, "best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"Saved BEST checkpoint -> {best_path}")

        # Save epoch checkpoint
        epoch_path = os.path.join(ckpt_dir, f"epoch_{epoch}.pt")
        torch.save(model.state_dict(), epoch_path)
        print(f"Saved checkpoint -> {epoch_path}")

        # Early stopping check - if still at 0.97 after warmup, something's wrong
        if epoch == WARMUP_EPOCHS + 5 and val_loss > 0.95:
            print("\n[WARNING] Loss still high after warmup+5 epochs!")
            print("The model may not be learning. Check gradients.")

    return best_val


EXPERIMENTS = {
    "transformer_matched_v4_frob": {
        "arch": "transformer",
        "loss": "frob",
        "lr": 1e-3,  # Higher LR for larger model
        "create_model": lambda: TransformerAutoencoderMatchedV4(
            loss_fn=FrobeniusFidelityLoss()
        )
    },
    "transformer_matched_v4_physics": {
        "arch": "transformer",
        "loss": "physics",
        "lr": 1e-3,
        "create_model": lambda: TransformerAutoencoderMatchedV4(
            loss_fn=CompositePhysicsTotalLoss()
        )
    },
}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and split dataset
    chunks = load_chunks("dataset_smaller")
    train_chunks, val_chunks, test_chunks = split_chunks(chunks)

    # Train both variants
    for name, config in EXPERIMENTS.items():
        train_v4_experiment(name, config, train_chunks, val_chunks, device)

    print("\nDone! Run the following for evaluation:")
    print("  python eval_transformer_matched_v4_uhlmann.py")
    print("  python eval_transformer_matched_v4_per_noise_cell.py")


if __name__ == "__main__":
    main()