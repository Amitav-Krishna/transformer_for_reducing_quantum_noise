"""
Test maximum batch size for 8-qubit transformer on GPU.
Binary search to find the largest batch size that fits in memory.
"""

import torch
import torch.nn as nn
import gc

# Add parent to path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_16.models.transformer_8qubit import HierarchicalTransformer8Qubit
from train_16.models.mlp_8qubit import HierarchicalMLP8Qubit
from losses.frob import FrobeniusFidelityLoss


def test_batch_size(
    model_class, model_name, batch_size, matrix_size, dtype=torch.float64
):
    """Test if a batch size fits in GPU memory with forward + backward pass."""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    try:
        model = model_class(loss_fn=FrobeniusFidelityLoss())
        model = model.to("cuda").to(dtype)

        # Create dummy data
        x = torch.randn(
            batch_size, 2, matrix_size, matrix_size, dtype=dtype, device="cuda"
        )
        y = torch.randn(
            batch_size, 2, matrix_size, matrix_size, dtype=dtype, device="cuda"
        )

        # Forward pass
        pred = model(x)
        loss = model.compute_loss(pred, y)

        # Backward pass
        loss.backward()

        peak_mem = torch.cuda.max_memory_allocated() / 1e9  # GB

        # Cleanup
        del model, x, y, pred, loss
        torch.cuda.empty_cache()
        gc.collect()

        return True, peak_mem
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            gc.collect()
            return False, 0
        raise


def find_max_batch_size(model_class, model_name, matrix_size, dtype=torch.float64):
    """Binary search for max batch size."""
    print(f"\n{'=' * 60}")
    print(f"Testing {model_name} (matrix: {matrix_size}x{matrix_size}, dtype: {dtype})")
    print(f"{'=' * 60}")

    # Start with powers of 2
    low, high = 1, 256
    best = 1
    best_mem = 0

    # First find upper bound
    batch = 1
    while batch <= 256:
        success, mem = test_batch_size(
            model_class, model_name, batch, matrix_size, dtype
        )
        if success:
            print(f"  batch_size={batch:3d}: OK ({mem:.2f} GB)")
            best = batch
            best_mem = mem
            batch *= 2
        else:
            print(f"  batch_size={batch:3d}: OOM")
            high = batch
            break
    else:
        high = 512

    # Binary search between best and high
    low = best
    while low < high - 1:
        mid = (low + high) // 2
        success, mem = test_batch_size(model_class, model_name, mid, matrix_size, dtype)
        if success:
            print(f"  batch_size={mid:3d}: OK ({mem:.2f} GB)")
            low = mid
            best = mid
            best_mem = mem
        else:
            print(f"  batch_size={mid:3d}: OOM")
            high = mid

    print(f"\n  MAX BATCH SIZE: {best} (peak memory: {best_mem:.2f} GB)")
    return best, best_mem


def main():
    if not torch.cuda.is_available():
        print("No GPU available!")
        return

    device_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {device_name}")
    print(f"Total memory: {total_mem:.1f} GB")

    # Test 8-qubit models (the limiting factor)
    results = {}

    # 8-qubit Transformer
    max_bs, mem = find_max_batch_size(
        HierarchicalTransformer8Qubit, "transformer_8qubit", matrix_size=256
    )
    results["transformer_8qubit"] = (max_bs, mem)

    # 8-qubit MLP
    max_bs, mem = find_max_batch_size(
        HierarchicalMLP8Qubit, "mlp_8qubit", matrix_size=256
    )
    results["mlp_8qubit"] = (max_bs, mem)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, (bs, mem) in results.items():
        print(f"  {name}: max_batch_size={bs}, peak_mem={mem:.2f} GB")

    min_bs = min(bs for bs, _ in results.values())
    print(f"\n  RECOMMENDED BATCH SIZE: {min_bs}")
    print(f"  (Use this for all models for fair comparison)")


if __name__ == "__main__":
    main()
