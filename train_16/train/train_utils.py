"""
Shared training utilities for train_16 experiments.

All experiments use:
- float64 precision
- batch_size=4
- AdamW optimizer with lr=3e-4, weight_decay=1e-4
- 100 epochs
- VRAM logging
"""

import os
import csv
import time
import torch
from torch.utils.data import DataLoader

# Add parent to path for imports
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_loop.dataset.load_chunks import load_chunks
from training_loop.dataset.split_chunks import split_chunks
from training_loop.dataset.ChunkDataset import ChunkDataset
from training_loop.dataset.StreamingChunkDataset import (
    StreamingChunkDataset,
    get_chunk_files,
    split_chunk_files,
)


# ============================================================================
# Constants
# ============================================================================

BATCH_SIZE = 256  # Max ~511 for 8-qubit on RTX A4000, using 256 for safety
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 100
DTYPE = torch.float64


# ============================================================================
# VRAM Logging
# ============================================================================


def get_gpu_memory_mb():
    """Get current and peak GPU memory usage in MB."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (
        torch.cuda.memory_allocated() / 1e6,
        torch.cuda.max_memory_allocated() / 1e6,
    )


def reset_peak_memory():
    """Reset peak memory stats for next measurement."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ============================================================================
# Data Loading (5-qubit)
# ============================================================================


def load_5qubit_data():
    """Load 5-qubit float64 dataset, returns (train_chunks, val_chunks, test_chunks)."""
    # Prefer float64 dataset, fall back to original if not found
    if os.path.exists("/workspace/dataset_5qubit_float64"):
        dataset_dir = "/workspace/dataset_5qubit_float64"
    elif os.path.exists("dataset_5qubit_float64"):
        dataset_dir = "dataset_5qubit_float64"
    elif os.path.exists("/workspace/dataset_smaller"):
        dataset_dir = "/workspace/dataset_smaller"
    else:
        dataset_dir = "dataset_smaller"

    print(f"Loading 5-qubit dataset from {dataset_dir}...")
    chunks = load_chunks(dataset_dir)

    if len(chunks) == 0:
        raise RuntimeError(f"No chunks found in {dataset_dir}")

    train_chunks, val_chunks, test_chunks = split_chunks(chunks, 0.8, 0.1, seed=42)

    train_samples = sum(X.shape[0] for X, Y in train_chunks)
    val_samples = sum(X.shape[0] for X, Y in val_chunks)
    test_samples = sum(X.shape[0] for X, Y in test_chunks)

    print(f"  Train: {len(train_chunks)} chunks, {train_samples:,} samples")
    print(f"  Val: {len(val_chunks)} chunks, {val_samples:,} samples")
    print(f"  Test: {len(test_chunks)} chunks, {test_samples:,} samples")

    return train_chunks, val_chunks, test_chunks


# ============================================================================
# Evaluation
# ============================================================================


@torch.no_grad()
def evaluate(model, chunks, device, batch_size=BATCH_SIZE):
    """Evaluate model on validation/test chunks."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for X, Y in chunks:
        ds = ChunkDataset(X, Y, mode="transformer")
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = model.compute_loss(pred, y)
            total_loss += loss.item() * x.shape[0]
            total_samples += x.shape[0]

    model.train()
    return total_loss / total_samples


# ============================================================================
# CSV Logging
# ============================================================================


class CSVLogger:
    """Simple CSV logger for training metrics."""

    def __init__(self, filepath, headers):
        self.filepath = filepath
        self.file = open(filepath, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(headers)
        self.file.flush()

    def log(self, *values):
        self.writer.writerow(values)
        self.file.flush()

    def close(self):
        self.file.close()


# ============================================================================
# Training Loop
# ============================================================================


def train_5qubit_model(
    model,
    model_name: str,
    train_chunks,
    val_chunks,
    device,
    checkpoint_dir: str,
    csv_dir: str,
):
    """
    Train a 5-qubit model with full logging.

    Args:
        model: The model to train (already on device)
        model_name: Name for logging (e.g., "transformer_5qubit")
        train_chunks: Training data chunks
        val_chunks: Validation data chunks
        device: torch device
        checkpoint_dir: Directory to save checkpoints
        csv_dir: Directory to save CSV logs
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Setup loggers
    train_logger = CSVLogger(
        os.path.join(csv_dir, f"{model_name}_train.csv"), ["epoch", "batch", "loss"]
    )
    val_logger = CSVLogger(
        os.path.join(csv_dir, f"{model_name}_val.csv"), ["epoch", "loss"]
    )
    timing_logger = CSVLogger(
        os.path.join(csv_dir, f"{model_name}_timing.csv"),
        ["epoch", "train_time_s", "val_time_s", "total_time_s"],
    )
    vram_logger = CSVLogger(
        os.path.join(csv_dir, f"{model_name}_vram.csv"),
        ["epoch", "current_mb", "peak_mb"],
    )

    best_val = float("inf")
    best_checkpoint_path = os.path.join(checkpoint_dir, "best.pt")

    print(f"\nTraining {model_name} for {EPOCHS} epochs...")
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"  Logs: {csv_dir}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print()

    total_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        reset_peak_memory()

        # Training
        train_start = time.time()
        model.train()
        epoch_losses = []
        batch_idx = 0

        for X, Y in train_chunks:
            ds = ChunkDataset(X, Y, mode="transformer")
            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

            for x, y in loader:
                x, y = x.to(device), y.to(device)

                pred = model(x)
                loss = model.compute_loss(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())
                batch_idx += 1

        train_time = time.time() - train_start
        avg_train_loss = sum(epoch_losses) / len(epoch_losses)

        # Log last batch loss
        train_logger.log(epoch, batch_idx, avg_train_loss)

        # Validation
        val_start = time.time()
        val_loss = evaluate(model, val_chunks, device, BATCH_SIZE)
        val_time = time.time() - val_start

        total_time = time.time() - epoch_start

        # Log metrics
        val_logger.log(epoch, val_loss)
        timing_logger.log(epoch, train_time, val_time, total_time)

        current_vram, peak_vram = get_gpu_memory_mb()
        vram_logger.log(epoch, current_vram, peak_vram)

        # Print progress
        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"Train: {avg_train_loss:.6f} | Val: {val_loss:.6f} | "
            f"Time: {total_time:.1f}s | VRAM: {peak_vram:.0f}MB"
        )

        # Save best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"  -> Saved best checkpoint (val_loss={val_loss:.6f})")

        # Periodic checkpoint
        if epoch % 10 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt"),
            )

    # Cleanup
    train_logger.close()
    val_logger.close()
    timing_logger.close()
    vram_logger.close()

    total_elapsed = time.time() - total_start
    print(f"\nTraining complete!")
    print(f"  Total time: {total_elapsed / 3600:.2f} hours")
    print(f"  Best val loss: {best_val:.6f}")
    print(f"  Best checkpoint: {best_checkpoint_path}")

    return best_val, best_checkpoint_path


# ============================================================================
# Data Loading (8-qubit streaming)
# ============================================================================


def load_8qubit_file_lists():
    """Load 8-qubit float64 chunk file lists for streaming, returns (train_files, val_files, test_files)."""
    # Prefer float64 dataset, fall back to original if not found
    if os.path.exists("/workspace/dataset_8qubit_float64"):
        dataset_dir = "/workspace/dataset_8qubit_float64"
    elif os.path.exists("dataset_8qubit_float64"):
        dataset_dir = "dataset_8qubit_float64"
    elif os.path.exists("/workspace/dataset_8qubit"):
        dataset_dir = "/workspace/dataset_8qubit"
    else:
        dataset_dir = "dataset_8qubit"

    print(f"Loading 8-qubit dataset file list from {dataset_dir}...")
    chunk_files = get_chunk_files(dataset_dir)

    if len(chunk_files) == 0:
        raise RuntimeError(f"No chunk files found in {dataset_dir}")

    train_files, val_files, test_files = split_chunk_files(chunk_files, seed=42)

    # Estimate samples (1000 per chunk)
    samples_per_chunk = 1000
    train_samples = len(train_files) * samples_per_chunk
    val_samples = len(val_files) * samples_per_chunk
    test_samples = len(test_files) * samples_per_chunk

    print(f"  Train: {len(train_files)} chunks, ~{train_samples:,} samples")
    print(f"  Val: {len(val_files)} chunks, ~{val_samples:,} samples")
    print(f"  Test: {len(test_files)} chunks, ~{test_samples:,} samples")

    return train_files, val_files, test_files


# ============================================================================
# Evaluation (8-qubit streaming)
# ============================================================================


@torch.no_grad()
def evaluate_streaming(model, val_files, device, batch_size=BATCH_SIZE):
    """Evaluate model on validation files using streaming."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for fpath in val_files:
        blob = torch.load(fpath, weights_only=False)
        X = blob["X"]  # Keep original dtype (float64)
        Y = blob["Y"]
        del blob

        n = len(X)
        for i in range(0, n, batch_size):
            x = X[i : i + batch_size].permute(0, 3, 1, 2).to(device)  # (B, 2, H, W)
            y = Y[i : i + batch_size].permute(0, 3, 1, 2).to(device)

            pred = model(x)
            loss = model.compute_loss(pred, y)
            total_loss += loss.item() * x.shape[0]
            total_samples += x.shape[0]

        del X, Y

    model.train()
    return total_loss / total_samples


# ============================================================================
# Training Loop (8-qubit streaming)
# ============================================================================


def train_8qubit_model(
    model,
    model_name: str,
    train_files,
    val_files,
    device,
    checkpoint_dir: str,
    csv_dir: str,
):
    """
    Train an 8-qubit model with streaming data loading.

    Args:
        model: The model to train (already on device)
        model_name: Name for logging (e.g., "transformer_8qubit")
        train_files: List of training chunk file paths
        val_files: List of validation chunk file paths
        device: torch device
        checkpoint_dir: Directory to save checkpoints
        csv_dir: Directory to save CSV logs
    """
    from tqdm import tqdm

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Setup loggers
    train_logger = CSVLogger(
        os.path.join(csv_dir, f"{model_name}_train.csv"), ["epoch", "batch", "loss"]
    )
    val_logger = CSVLogger(
        os.path.join(csv_dir, f"{model_name}_val.csv"), ["epoch", "loss"]
    )
    timing_logger = CSVLogger(
        os.path.join(csv_dir, f"{model_name}_timing.csv"),
        ["epoch", "train_time_s", "val_time_s", "total_time_s"],
    )
    vram_logger = CSVLogger(
        os.path.join(csv_dir, f"{model_name}_vram.csv"),
        ["epoch", "current_mb", "peak_mb"],
    )

    best_val = float("inf")
    best_checkpoint_path = os.path.join(checkpoint_dir, "best.pt")
    samples_per_chunk = 1000

    print(f"\nTraining {model_name} for {EPOCHS} epochs (STREAMING MODE)...")
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"  Logs: {csv_dir}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print()

    total_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        reset_peak_memory()

        # Training with streaming
        train_start = time.time()
        model.train()
        epoch_losses = []

        train_dataset = StreamingChunkDataset(
            train_files, shuffle=True, samples_per_chunk=samples_per_chunk
        )
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0)

        batch_idx = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = model.compute_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            batch_idx += 1

        train_time = time.time() - train_start
        avg_train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0

        # Log
        train_logger.log(epoch, batch_idx, avg_train_loss)

        # Validation
        val_start = time.time()
        val_loss = evaluate_streaming(model, val_files, device, BATCH_SIZE)
        val_time = time.time() - val_start

        total_time = time.time() - epoch_start

        # Log metrics
        val_logger.log(epoch, val_loss)
        timing_logger.log(epoch, train_time, val_time, total_time)

        current_vram, peak_vram = get_gpu_memory_mb()
        vram_logger.log(epoch, current_vram, peak_vram)

        # Print progress
        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"Train: {avg_train_loss:.6f} | Val: {val_loss:.6f} | "
            f"Time: {total_time:.1f}s | VRAM: {peak_vram:.0f}MB"
        )

        # Save best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"  -> Saved best checkpoint (val_loss={val_loss:.6f})")

        # Periodic checkpoint
        if epoch % 10 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt"),
            )

    # Cleanup
    train_logger.close()
    val_logger.close()
    timing_logger.close()
    vram_logger.close()

    total_elapsed = time.time() - total_start
    print(f"\nTraining complete!")
    print(f"  Total time: {total_elapsed / 3600:.2f} hours")
    print(f"  Best val loss: {best_val:.6f}")
    print(f"  Best checkpoint: {best_checkpoint_path}")

    return best_val, best_checkpoint_path
