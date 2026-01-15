"""
train_16: Float64 End-to-End Training Pipeline

This module contains the complete float64 precision training pipeline for
quantum density matrix denoising. All models, data, and evaluations use
double precision (float64) to ensure numerical stability during
eigendecomposition for Uhlmann fidelity computation.

Experiments:
-----------
5-qubit (32x32 matrices, 4x4 patches -> 64 tokens):
  - Transformer: ~1.09M params, 4+4 layers, 8 heads
  - MLP Matched: ~1.09M params, 4+4 layers (same params as Transformer)
  - MLP Wide: ~5.2M params, 4+4 layers (capacity control)
  - MLP Deep: ~2.15M params, 8+8 layers (depth control)

8-qubit (256x256 matrices, 32x32 patches -> 64 tokens):
  - Transformer: ~1.61M params, 4+4 layers, 8 heads
  - MLP Matched: ~1.61M params, 4+4 layers

Key Design Decisions:
--------------------
1. All models use dtype=torch.float64 for weights
2. All data stored/loaded as float64
3. Batch size = 4 for all experiments (consistency)
4. Hierarchical tokenization: constant 64 tokens regardless of qubit count
5. VRAM logging for scalability arguments
6. Uhlmann fidelity with skip-and-count failure tracking
7. Self-check: F(clean, clean) = 1.0 verification

Output:
------
- checkpoints_16/: Model checkpoints
- csvs_16/: Training logs, timing, VRAM, Uhlmann fidelity results
"""
