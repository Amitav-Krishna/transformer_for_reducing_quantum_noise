# Comparing Machine Learning Strategies for Quantum Noise Reduction

This project compares neural network autoencoders for density-matrix denoising across multiple quantum noise channels.

**Paper v2 (completed):** CNN vs Transformer comparison → Transformer wins (0.95 vs 0.30 fidelity)

**Paper v3 (completed):** MLP vs Transformer comparison

**Paper v4 (in progress):** Hierarchical models + scaling (5-qubit → 8-qubit) with capacity/depth ablations
- v3/v4: Cholesky-constrained outputs → **FAILED** (collapsed to maximally mixed state)
- v5: Row-based tokenization → **FAILED** (stuck at 0.968 val loss)
- v6: Element-wise tokenization (~120k params each) → MLP destroyed info (0.038 fidelity < baseline)
- v7: Residual MLP → **SUCCESS** (intermediate results)
- v8: Final training on complete dataset → **Baseline: 0.12, MLP: 0.17 (1.4×), Transformer: 0.28 (2.3×)**
- v9: Wide MLP capacity control experiment → **1M param MLP still underperforms 119k Transformer**
- v11: **Pauli representation** → Removes spatial structure, tests algebraic vs spatial learning. **Transformer achieves 0.33 vs MLP 0.19** (1.7× gap)

## Project Structure

```
.
├── paper_v2.org            # Paper v2 (Org-mode source) - CNN vs Transformer
├── paper_v3.org            # Paper v3 (Org-mode source) - MLP vs Transformer (final: residual MLP)
├── paper_v3.pdf            # Paper v3 compiled PDF
├── paper_v4.org            # Paper v4 (Org-mode source) - Hierarchical + scaling (in progress)
├── references.bib          # Bibliography
│
├── dataset_smaller/        # Chunked training dataset (100k samples, float32)
├── dataset_5qubit_float64/ # Float64 5-qubit dataset (100k samples) - train_16
├── dataset_8qubit_float64/ # Float64 8-qubit dataset (100k samples) - train_16
├── dataset.pt              # Original full dataset (1M samples)
│
├── models/                 # Original model architectures (CNN, Transformer)
├── models_3/               # Cholesky-output models (MLP, Transformer) - FAILED
├── models_4/               # Unconstrained MLP/Transformer (global CLS) - FAILED
├── models_5/               # Row-based tokenization - FAILED
├── train_v6/               # Element-wise tokenization (~120k params) - Transformer works, MLP destroys info
├── train_v7/               # Residual MLP (~117k params) - fixes MLP info destruction
├── train_v8/               # **FINAL**: Residual MLP + Transformer on full dataset
├── train_v9/               # Wide MLP capacity control (~1M params)
├── train_11/               # **PAULI**: Representation without spatial structure
├── train_16/               # **CURRENT**: Float64 end-to-end pipeline (5-qubit + 8-qubit)
├── losses/                 # Loss functions
├── training_loop/          # Training infrastructure
│
├── checkpoints_2/          # Paper v2 checkpoints (CNN/Transformer)
├── checkpoints_3/          # Paper v3 checkpoints (MLP/Transformer Cholesky)
├── csvs_2/                 # Paper v2 experiment results
├── csvs_3/                 # Paper v3 experiment results (CANONICAL - synced from runpod)
├── csvs_3_new/             # Backup copy of csvs_3 (nested structure, same data)
├── csvs_3_old/             # Older training run (53/40 epochs) - superseded
├── figures/                # Generated figures for paper
│
├── train_models.py         # Train CNN/Transformer (paper v2)
├── train_cholesky.py       # Train MLP/Transformer Cholesky (paper v3)
├── train_transformer_cholesky_v2.py  # NEW: Train FIXED transformer only
└── eval_*.py               # Various evaluation scripts
```

---

## Paper v3: Cholesky Models (NEW)

### Why Cholesky?

Paper v2 showed CNNs fail at density matrix denoising. But CNNs are a poor baseline - they're designed for images, not quantum states. Paper v3 uses MLPs as a fairer comparison since they don't impose spatial locality bias.

Both models use **Cholesky output layers** that guarantee valid density matrices (Hermitian, trace=1, PSD) by construction.

### Models (`models_3/`)

| File | Description |
|------|-------------|
| `cholesky_output.py` | Cholesky decomposition layer: params → L → ρ = LL†/Tr(LL†) |
| `mlp_cholesky.py` | MLP with Cholesky output (~427k params) |
| `transformer_cholesky.py` | Transformer with Cholesky output (~492k params) - **FIXED VERSION** |

### Training Scripts

| Script | Description |
|--------|-------------|
| `train_cholesky.py` | Trains both MLP and Transformer Cholesky models |
| `train_transformer_cholesky_v2.py` | **NEW:** Trains only the FIXED transformer (after collapse diagnosis) |

### Evaluation Scripts

| Script | Description |
|--------|-------------|
| `eval_cholesky_uhlmann.py` | Evaluate Cholesky models on Uhlmann fidelity |
| `eval_cholesky_per_noise_cell.py` | Evaluate by noise type/level |
| `test_cholesky_models.py` | Verify outputs are valid density matrices |

### Results Status

**MLP Cholesky:** 96 epochs (early stopped), val loss 0.799, fidelity 0.03-0.07 (learning but poor)

**Transformer Cholesky v1:** 24 epochs, val loss 0.823, fidelity **0.031 constant** - **COLLAPSED!**
- Diagnosed: degenerate decoder + per-row projection prevented global Cholesky coordination
- Output was maximally mixed state (I/32)

**Transformer Cholesky v2:** Training pending (fixed architecture)

### The CSV Mess Explained

Multiple training runs created confusing directory structure:

| Directory | Contents | Status |
|-----------|----------|--------|
| `csvs_3/` | **CANONICAL** - Latest run (MLP 96 epochs, Transformer 24 epochs) | Use this |
| `csvs_3_new/csvs_3/` | Same data as csvs_3/ (nested copy from rsync) | Backup |
| `csvs_3_old/` | Older run (MLP 53 epochs, Transformer 40 epochs) | Superseded |

### Debugging Materials

| Directory | Description |
|-----------|-------------|
| `debug_transformer_collapse/` | Diagnostic scripts and analysis of why transformer collapsed |
| `debug_transformer_collapse/DIAGNOSIS.md` | Full technical analysis |
| `debug_transformer_collapse/QUICK_SUMMARY.txt` | TL;DR of the problem |

---

## Paper v2: CNN vs Transformer (Original)

### Scripts

#### Training & Evaluation

| Script | Description |
|--------|-------------|
| `train_models.py` | Main entry point. Trains original 4 model configurations (CNN/Transformer x Frobenius/Physics loss) |
| `train_transformer_matched_params.py` | Trains capacity-matched Transformer v1 (~751k params) - **has bottleneck bug, see Appendix** |
| `train_transformer_matched_v2.py` | Trains capacity-matched Transformer v2 (~757k params) - **still has issues, see v3** |
| `train_transformer_matched_v3.py` | Trains capacity-matched Transformer v3 (~741k params) with proper FFN ratio and lower LR |
| `train_transformer_matched_v4.py` | Trains capacity-matched Transformer v4 (~752k params) with warmup, gradient clipping, Pre-LN |
| `eval_models_on_uhlmann.py` | Evaluates trained models using Uhlmann fidelity on test set |
| `eval_per_noise_cell.py` | Evaluates models broken down by noise type and noise level |
| `eval_transformer_matched_v2_uhlmann.py` | Evaluates v2 transformer on Uhlmann fidelity |
| `eval_transformer_matched_v2_per_noise_cell.py` | Evaluates v2 transformer per noise type/level |
| `eval_transformer_matched_v3_uhlmann.py` | Evaluates v3 transformer on Uhlmann fidelity |
| `eval_transformer_matched_v3_per_noise_cell.py` | Evaluates v3 transformer per noise type/level |
| `eval_transformer_matched_v4_uhlmann.py` | Evaluates v4 transformer on Uhlmann fidelity |
| `eval_transformer_matched_v4_per_noise_cell.py` | Evaluates v4 transformer per noise type/level |
| `eval_models.py` | General evaluation utilities |
| `ground_truth_fidelity.py` | Computes baseline Uhlmann fidelity (noisy vs clean) - **source of the 0.11 baseline** |
| `generate_dataset.py` | Generates synthetic quantum circuits with noise using Cirq |

### Figure Generation

| Script | Description |
|--------|-------------|
| `figures/baseline_comparison_charts.py` | Generates baseline heatmap, improvement heatmaps, and bar charts |
| `figures/transformer_architecture.py` | Generates transformer architecture diagram |
| `figures/uhlmann_bar_chart.py` | Generates Uhlmann fidelity comparison bar chart |

---

## Models (`models/`)

| File | Description |
|------|-------------|
| `cnn.py` | CNN Autoencoder (748,898 params) - treats density matrix as 2-channel (real/imag) image |
| `transformer.py` | Transformer Autoencoder (119,506 params) - treats density matrix as 1024 tokens |
| `transformer_matched_params.py` | Capacity-matched Transformer v1 (751,186 params) - **broken bottleneck 96→16→96** |
| `transformer_matched_v2.py` | Capacity-matched Transformer v2 (757,362 params) - fixed bottleneck but wrong FFN ratio |
| `transformer_matched_v3.py` | Capacity-matched Transformer v3 (740,736 params) - proper 4x FFN ratio, 50% bottleneck |
| `transformer_matched_v4.py` | Capacity-matched Transformer v4 (~752k params) - Pre-LN, depth-scaled init, 2x FFN, warmup |

---

## Utility Scripts (`scripts/`)

| File | Description |
|------|-------------|
| `count_params.py` | Counts and compares parameters across all model architectures |

---

## Loss Functions (`losses/`)

| File | Description |
|------|-------------|
| `frob.py` | Normalized Frobenius fidelity loss (cosine similarity for matrices) |
| `physics.py` | Physics constraints: Hermiticity, unit trace, positive semi-definiteness |
| `total_physics_loss.py` | Composite loss combining Frobenius + physics penalties |

---

## Training Infrastructure (`training_loop/`)

| File | Description |
|------|-------------|
| `train_single_experiment.py` | Training loop for a single model configuration |
| `evaluate_all_on_test.py` | Evaluation across all checkpoint epochs |
| `fidelity.py` | Fidelity computation utilities |
| `dataset/ChunkDataset.py` | PyTorch Dataset wrapper for chunked data |
| `dataset/load_chunks.py` | Loads dataset chunks with optional metadata |
| `dataset/split_chunks.py` | Train/val/test split (80/10/10) |
| `dataset/csv_logger.py` | Logging utilities for training metrics |

---

## Data

### Dataset

| Path | Description |
|------|-------------|
| `dataset_smaller/` | Chunked dataset: 100k (noisy, clean) density matrix pairs |
| `dataset.pt` | Original 1M sample dataset (not used in final experiments) |

**Dataset format**: Each sample is a 32x32 complex density matrix stored as 2 real channels (real/imag). Noise types: depolarizing, amplitude damping, phase damping, bit-flip, mixed. Noise levels: 0.05, 0.10, 0.15, 0.20.

### Chunk Data Structure

Each `.pt` file in `dataset_smaller/` is a dict with the following structure:

```python
blob = torch.load("dataset_smaller/bitflip_0.2_part75050.pt")
blob.keys()  # dict_keys(['X', 'Y', 'meta'])

# X: noisy density matrices, shape (N, 2, 32, 32) - real/imag channels
# Y: clean density matrices, shape (N, 2, 32, 32) - real/imag channels
# meta: list of N dicts, one per sample

blob["meta"][0]  # {'noise_type': 'bitflip', 'noise_level': 0.2, 'depth': 7}
```

**Loading functions** (`training_loop/dataset/load_chunks.py`):
- `load_chunks(dir)` → returns list of `(X, Y)` tuples (no metadata)
- `load_chunks_with_metadata(dir)` → returns list of full blob dicts (with metadata)

**Splitting** (`training_loop/dataset/split_chunks.py`):
```python
from training_loop.dataset.split_chunks import split_chunks

# Returns split chunks
train, val, test = split_chunks(chunks, 0.8, 0.1, seed=42)

# Or return indices only (for use with metadata)
train_idx, val_idx, test_idx = split_chunks(chunks, 0.8, 0.1, seed=42, return_indices=True)
```

### Model Checkpoints

#### Paper v2 Checkpoints (`checkpoints_2/`)

| Path | Description |
|------|-------------|
| `checkpoints_2/cnn_frob/` | CNN + Frobenius loss (100 epochs + `best.pt`) |
| `checkpoints_2/cnn_physics/` | CNN + Physics loss (100 epochs + `best.pt`) |
| `checkpoints_2/transformer_frob/` | Transformer + Frobenius loss (100 epochs + `best.pt`) |
| `checkpoints_2/transformer_physics/` | Transformer + Physics loss (100 epochs + `best.pt`) |
| `checkpoints_2/transformer_matched_frob/` | Capacity-matched Transformer (~751k params) + Frobenius loss |
| `checkpoints_2/transformer_matched_physics/` | Capacity-matched Transformer (~751k params) + Physics loss |

#### Paper v3 Checkpoints (`checkpoints_3/`)

| Path | Description |
|------|-------------|
| `checkpoints_3/mlp_cholesky/` | MLP Cholesky (96 epochs + `best.pt`) |
| `checkpoints_3/transformer_cholesky/` | Transformer Cholesky v1 - **COLLAPSED** (24 epochs + `best.pt`) |
| `checkpoints_3/transformer_cholesky_v2/` | Transformer Cholesky v2 - **PENDING** (fixed architecture) |

#### Other Checkpoint Directories

| Path | Description |
|------|-------------|
| `checkpoints_3_runpod/` | Copy of checkpoints_3/ synced from runpod (same data) |
| `checkpoints/` | Legacy checkpoints from earlier experiments |
| `legacy_models_checkpoints/` | Older model checkpoints |

---

## Results (`csvs_2/`)

### Training Logs

| File | Description |
|------|-------------|
| `cnn_frob.csv` | Full training log: epoch, batch, train_loss, val_loss |
| `cnn_physics.csv` | CNN Physics training log |
| `transformer_frob.csv` | Transformer Frobenius training log |
| `transformer_physics.csv` | Transformer Physics training log |
| `transformer_matched_frob.csv` | Capacity-matched Transformer + Frobenius training log |
| `transformer_matched_physics.csv` | Capacity-matched Transformer + Physics training log |

### Test Results

| File | Description |
|------|-------------|
| `cnn_frob_test.csv` | Per-epoch test metrics |
| `cnn_physics_test.csv` | CNN Physics test metrics |
| `transformer_frob_test.csv` | Transformer Frobenius test metrics |
| `transformer_physics_test.csv` | Transformer Physics test metrics |

### Aggregated Results

| Path | Description |
|------|-------------|
| `csvs_2/mean_train/` | Mean training loss per epoch for each model |
| `csvs_2/noise_cells/` | **Uhlmann fidelity by noise type and level** |
| `csvs_2/uhlmann_ground_truth/` | **Baseline noisy-vs-clean fidelity (source of 0.11)** |

### Key Files for Paper Claims

| File | Paper Claim |
|------|-------------|
| `csvs_2/uhlmann_ground_truth/noisy_vs_clean_test_uhlmann.csv` | Baseline fidelity 0.11 (raw per-sample data) |
| `csvs_2/noise_cells/ground_truth_noise_cells.csv` | Baseline fidelity by noise type/level (aggregated) |
| `csvs_2/noise_cells/transformer_frob_noise_cells.csv` | Transformer Frob achieves 0.95 fidelity |
| `csvs_2/noise_cells/cnn_frob_noise_cells.csv` | CNN Frob achieves 0.30 fidelity |
| `csvs_2/noise_cells/cnn_physics_noise_cells.csv` | CNN Physics achieves 0.06 fidelity |
| `csvs_2/noise_cells/transformer_physics_noise_cells.csv` | Transformer Physics achieves 0.33 fidelity |

---

## Results (`csvs_3/`) - Paper v3 Cholesky Experiments

### Training Logs

| File | Description |
|------|-------------|
| `mlp_cholesky.csv` | MLP Cholesky training log (96 epochs, early stopped) |
| `transformer_cholesky.csv` | Transformer Cholesky v1 training log (24 epochs, early stopped) - **COLLAPSED** |
| `transformer_cholesky_v2.csv` | Transformer Cholesky v2 training log - **PENDING** |

### Per-Noise-Cell Results (`csvs_3/noise_cells/`)

| File | Description |
|------|-------------|
| `mlp_cholesky_noise_cells.csv` | MLP fidelity by noise type/level (0.03-0.07, learning) |
| `transformer_cholesky_noise_cells.csv` | Transformer v1 fidelity - **constant 0.031** (collapsed to I/32) |

### Directory Variants Explained

| Directory | What It Is |
|-----------|------------|
| `csvs_3/` | **USE THIS** - Canonical results synced from runpod |
| `csvs_3_new/` | Contains `csvs_3/` subdirectory - artifact of nested rsync |
| `csvs_3_new/csvs_3/` | Same data as `csvs_3/` (redundant copy) |
| `csvs_3_old/` | Earlier run with worse results (MLP 53ep/0.84 loss, Trans 40ep/0.86 loss) |

---

## Paper v3: Unconstrained Models (v4, v5, v6)

After Cholesky failed, we tried several unconstrained architectures:

### v4: Unconstrained with Global CLS Projection (`models_4/`)

- **MLP**: 2048→512→256→512→2048 (~1.3M params)
- **Transformer**: Row-based (32 tokens + CLS), global projection from CLS to 2048 outputs
- **Result**: MLP val loss 0.804, Transformer stuck at 0.825 - **FAILED**
- **Problem**: Single CLS token can't coordinate 2048 outputs

### v5: Row-based Tokenization with Per-Row Output (`models_5/`)

- **MLP**: Same as v4
- **Transformer**: 32 row tokens, each outputs 64 values (per-row projection)
- **Result**: Transformer stuck at 0.968 val loss - **FAILED**
- **Problem**: Row-level attention can't capture element-to-element correlations

### v6: Element-wise Tokenization (`train_v6/`)

- **MLP**: 2048→28→14→28→2048 (~117k params)
- **Transformer**: 1024 element tokens, each outputs 2 values (~119k params)
- **Result**: MLP Uhlmann fidelity **0.038** (worse than 0.068 baseline!), Transformer **0.172**
- **Problem**: MLP's severe bottleneck destroyed information faster than it could learn

| File | Description |
|------|-------------|
| `train_v6/mlp.py` | Small MLP (~117k params) - **destroys info** |
| `train_v6/transformer.py` | Element-wise Transformer (~119k params) |
| `train_v6/train.py` | Training script |
| `train_v6/eval.py` | Uhlmann fidelity evaluation |
| `train_v6/baseline_fidelity.py` | Computes baseline (noisy vs clean) Uhlmann fidelity |
| `train_v6/csvs_6/` | Training logs |
| `train_v6/checkpoints_6/` | Model checkpoints |

### v7: Residual MLP (`train_v7/`)

- **MLP**: Residual architecture: `output = input + correction(input)`, 2048→28→28→2048 (~117k params)
- **Key fix**: Skip connection preserves input information; output layer initialized to zero
- **Result**: Intermediate results, superseded by v8

| File | Description |
|------|-------------|
| `train_v7/mlp.py` | Residual MLP (~117k params) |
| `train_v7/train.py` | Training script (MLP only) |
| `train_v7/eval.py` | Uhlmann fidelity evaluation |
| `train_v7/eval_per_noise_cell.py` | Per noise-type/level evaluation for heatmaps |
| `train_v7/generate_val_loss_figure.py` | Standalone validation loss figure script |
| `train_v7/generate_heatmaps.py` | Standalone heatmap generation script |
| `train_v7/csvs_7/` | Training logs |
| `train_v7/checkpoints_7/` | Model checkpoints |

### v8: Final Models - **CURRENT** (`train_v8/`)

- **MLP**: Residual architecture from v7 (~117k params)
- **Transformer**: Element-wise tokenization from v6 (~119k params)
- **Dataset**: Complete dataset with all 5 noise types including phase_damping
- **Result**: Baseline 0.12, MLP **0.17** (1.4×), Transformer **0.28** (2.3×)

| File | Description |
|------|-------------|
| `train_v8/mlp.py` | Residual MLP (~117k params) |
| `train_v8/transformer.py` | Element-wise Transformer (~119k params) |
| `train_v8/train.py` | Training script for both models |
| `train_v8/eval_per_noise_cell.py` | Per noise-type/level Uhlmann fidelity evaluation |
| `train_v8/generate_val_loss_figure.py` | Validation loss overlay figure |
| `train_v8/generate_heatmaps.py` | Per-noise-cell heatmap generation |
| `train_v8/csvs_8/` | Training logs |
| `train_v8/csvs_8/noise_cells/` | Per-noise-cell Uhlmann fidelity CSVs |
| `train_v8/checkpoints_8/` | Model checkpoints (`best.pt` for each) |

### v9: Wide MLP Capacity Control (`train_v9/`)

Control experiment testing whether MLP underperformance is due to insufficient capacity.

- **Wide MLP**: 2048→512→256→512→2048 (~1M params) - 8× more params than Transformer
- **Transformer**: Same as v8 (~119k params)
- **Result**: Wide MLP still only achieves ~0.19 Uhlmann fidelity vs Transformer 0.28
- **Conclusion**: Capacity is not the bottleneck - architecture matters

| File | Description |
|------|-------------|
| `train_v9/mlp_wide.py` | Wide MLP (~1M params) |
| `train_v9/transformer.py` | Element-wise Transformer (~119k params) |
| `train_v9/train.py` | Training script (wd=1e-4) |
| `train_v9/train_1e-3.py` | Training script (wd=1e-3) |
| `train_v9/eval_per_noise_cell.py` | Per noise-type/level evaluation |
| `train_v9/csvs_9/` | Training logs |
| `train_v9/mlp_wide_wd1e-03/` | Checkpoints with weight decay 1e-3 |
| `train_v9/mlp_wide_wd1e-04/` | Checkpoints with weight decay 1e-4 |

### v11: Pauli Representation (`train_11/`)

Control experiment that removes spatial structure by converting density matrices to Pauli basis.

- **Key idea**: 32x32 density matrix → 1024 real Pauli coefficients (flat vector)
- **Purpose**: Test whether Transformer advantage comes from spatial attention or algebraic structure learning
- **MLP Pauli**: Simple MLP on 1024 coefficients (~1M params)
- **Transformer Pauli**: 1024 tokens (one per Pauli operator), learns algebraic correlations
- **Result**: Transformer 0.33 fidelity vs MLP 0.19 (1.7× gap persists without spatial structure)
- **Insight**: Attention heads learn to focus on physically-relevant Pauli operators (X/Y coherence operators get 90× mean attention)

| File | Description |
|------|-------------|
| `train_11/pauli_representation.py` | Density matrix ↔ Pauli basis conversion utilities |
| `train_11/mlp_pauli.py` | MLP on Pauli coefficients (~1M params) |
| `train_11/transformer_pauli.py` | Transformer on Pauli tokens (~119k params) |
| `train_11/pauli_loss.py` | Pauli Frobenius loss (cosine similarity on coefficients) |
| `train_11/train.py` | Training script for both Pauli models |
| `train_11/eval_uhlmann.py` | Evaluate Pauli models on Uhlmann fidelity |
| `train_11/csvs_11/` | Training logs |
| `train_11/checkpoints_11/` | Model checkpoints |
| `train_11/checkpoints_11/mlp_pauli_frob/` | MLP Pauli checkpoints |
| `train_11/checkpoints_11/transformer_pauli_frob/` | Transformer Pauli checkpoints |

### train_15: Hierarchical MLP Experiments (`train_15/`)

Patch-based MLP models for fair comparison with hierarchical Transformers. Uses identical patch embedding/unembedding as Transformers, replacing only the attention mechanism with MLP-Mixer style token mixing.

**Architecture**:
- **Patch embedding**: Same as Transformer (Conv2d with patch_size stride)
- **Token mixing**: MLP that mixes across patches (replaces attention)
- **Channel mixing**: Per-patch MLP (same as Transformer FFN)
- **Patch unembedding**: Same as Transformer

**Key difference**: Token mixing uses fixed learned weights (O(n)) instead of input-dependent attention (O(n²)).

#### Models

| File | Params | Description |
|------|--------|-------------|
| `mlp_hierarchical_5qubit.py` | 1,092,960 | 5-qubit MLP, matched to Transformer |
| `mlp_hierarchical_5qubit_wide.py` | 5,201,408 | 5-qubit wide MLP (capacity control) |
| `mlp_hierarchical_8qubit.py` | 1,611,072 | 8-qubit MLP, matched to Transformer |

#### Training Scripts

| File | Description |
|------|-------------|
| `train_5qubit.py` | Train 5-qubit hierarchical MLP (matched params) |
| `train_5qubit_wide.py` | Train 5-qubit wide MLP (~5.2M params, capacity control) |
| `train.py` | Train 8-qubit hierarchical MLP with streaming |
| `queue_mlp_experiments.sh` | Queue script to run all MLP experiments after Transformer |

#### Parameter Matching

| Model | 5-qubit | 8-qubit |
|-------|---------|---------|
| Hierarchical Transformer | 1,092,960 | 1,611,072 |
| Hierarchical MLP (matched) | 1,092,960 | 1,611,072 |
| Hierarchical MLP Wide | 5,201,408 | — |

**Why patch-based models have more params than element-wise**: The patch embedding layer scales with patch size². For 8-qubit (32×32 patches): `2 × 128 × 1024 = 262k` params just for embedding, vs negligible for element-wise.

---

## Figures (`figures/`)

### Training Curves

| File | Description |
|------|-------------|
| `*_train_loss.pdf` | Training loss vs epoch |
| `*_val_loss.pdf` | Validation loss vs epoch |
| `fidelity_log_combined.png` | All models' fidelity on log scale |

### Heatmaps (Fidelity by Noise Type/Level)

| File | Description |
|------|-------------|
| `baseline_noisy_vs_clean_heatmap.pdf` | Baseline fidelity (noisy input vs clean target) |
| `cnn_frob_noise_cells_heatmap.pdf` | CNN Frobenius fidelity heatmap |
| `cnn_physics_noise_cells_heatmap.pdf` | CNN Physics fidelity heatmap |
| `transformer_frob_noise_cells_heatmap.pdf` | Transformer Frobenius fidelity heatmap |
| `transformer_physics_noise_cells_heatmap.pdf` | Transformer Physics fidelity heatmap |

### Improvement Heatmaps (Model - Baseline)

| File | Description |
|------|-------------|
| `cnn_frob_improvement_heatmap.pdf` | CNN Frob improvement over baseline |
| `cnn_physics_improvement_heatmap.pdf` | CNN Physics improvement (shows degradation) |
| `transformer_frob_improvement_heatmap.pdf` | Transformer Frob improvement |
| `transformer_physics_improvement_heatmap.pdf` | Transformer Physics improvement |

### Bar Charts

| File | Description |
|------|-------------|
| `baseline_vs_models_bar.pdf` | Comparison of all models vs baseline |
| `uhlmann_cnn_vs_transformer.pdf` | CNN vs Transformer fidelity comparison |

### Architecture Diagrams

| File | Description |
|------|-------------|
| `transformer_architecture.pdf` | Transformer autoencoder architecture |

### Attention Analysis (Density Matrix Transformer)

| File | Description |
|------|-------------|
| `attention_maps.py` | Attention map visualization for entangled states (GHZ) |
| `attention_maps_simple.py` | Simplified bar chart attention visualization |
| `check_attention_values.py` | Script to compute attention statistics |
| `attention_from_queries_to_key0.pdf` | **Paper figure**: Which tokens attend to token 0 |
| `attention_to_keys.pdf` | Which keys receive most attention overall |
| `attention_maps/` | Directory with additional attention visualizations |

### Attention Analysis (Pauli Transformer)

| File | Description |
|------|-------------|
| `attention_maps_pauli.py` | Pauli attention visualization utilities and wrapper |
| `attention_pauli_paper.py` | Paper-quality Pauli attention figure generation |
| `pauli_attention_to_identity_paper.pdf` | **Paper figure**: X/Y coherence operators attend with 90× mean attention |
| `attention_maps_pauli/` | Directory with additional Pauli attention visualizations |

### Pauli Training Curves

| File | Description |
|------|-------------|
| `pauli_val_loss_chart.py` | Generates Pauli validation loss chart and rank correlation analysis |
| `pauli_val_loss.pdf` | **Paper figure**: MLP vs Transformer Pauli val loss (0.78 vs 0.39) |
| `pauli_frob_vs_uhlmann.pdf` | Scatter plot: Pauli Frobenius similarity vs Uhlmann fidelity (ρ=0.99) |

---

## Usage

### Training (Paper v2 - CNN/Transformer)

```bash
source venv/bin/activate
python train_models.py
```

### Training (Paper v3 - Cholesky Models)

```bash
source venv/bin/activate

# Train both MLP and Transformer Cholesky
python train_cholesky.py

# Train only the FIXED Transformer v2
python train_transformer_cholesky_v2.py
```

### Evaluation

```bash
# Paper v2 evaluation
python ground_truth_fidelity.py      # Baseline fidelity
python eval_models_on_uhlmann.py     # Uhlmann fidelity
python eval_per_noise_cell.py        # Per noise type/level

# Paper v3 evaluation (Cholesky models)
python eval_cholesky_uhlmann.py
python eval_cholesky_per_noise_cell.py
```

### Training (Pauli Representation - train_11)

```bash
source venv/bin/activate
python train_11/train.py
```

### Evaluation (Pauli Models)

```bash
python train_11/eval_uhlmann.py     # Uhlmann fidelity for Pauli models
```

### Generating Figures

```bash
python figures/baseline_comparison_charts.py
python figures/transformer_architecture.py

# Pauli figures
python figures/pauli_val_loss_chart.py      # Val loss + rank correlation
python figures/attention_pauli_paper.py     # Pauli attention visualization
python figures/attention_maps_pauli.py      # Full Pauli attention analysis
```

---

## Key Results

### Paper v2: CNN vs Transformer

| Model | Uhlmann Fidelity | Improvement over Baseline |
|-------|------------------|---------------------------|
| Baseline (noisy input) | 0.11 ± 0.15 | — |
| CNN Frobenius | 0.30 ± 0.28 | +0.19 |
| CNN Physics | 0.06 ± 0.05 | -0.05 (worse than baseline) |
| Transformer Frobenius | 0.95 ± 0.12 | +0.84 |
| Transformer Physics | 0.33 ± 0.33 | +0.22 |

### Paper v3: MLP vs Transformer (Final - v8)

| Model | Uhlmann Fidelity | Improvement |
|-------|------------------|-------------|
| Baseline (noisy input) | 0.12 ± 0.13 | — |
| MLP (residual) | **0.17 ± 0.17** | **1.4×** |
| Transformer | **0.28 ± 0.23** | **2.3×** |

**Key findings:**
- Transformer achieves 1.6× better fidelity than MLP despite similar parameter count (~120k)
- Residual architecture essential for MLP (non-residual destroyed information)
- Cholesky-constrained outputs and row-based tokenization both failed
- Element-wise tokenization allows modeling arbitrary pairwise correlations
- Attention analysis shows transformer learns to focus on entanglement-correlated elements

### Control Experiments: Capacity and Representation

#### v9: Wide MLP (Capacity Control)

| Model | Params | Uhlmann Fidelity |
|-------|--------|------------------|
| MLP Wide (wd=1e-4) | ~1M | 0.19 |
| MLP Wide (wd=1e-3) | ~1M | 0.19 |
| Transformer | ~119k | 0.28 |

**Conclusion**: 8× more parameters don't help MLP match Transformer. Architecture matters, not capacity.

#### v11: Pauli Representation (Spatial Structure Control)

| Model | Uhlmann Fidelity | Val Loss |
|-------|------------------|----------|
| Baseline (noisy) | 0.12 | — |
| MLP Pauli | **0.19** | 0.78 |
| Transformer Pauli | **0.33** | 0.39 |

**Conclusion**: Transformer advantage persists even without spatial structure. Attention learns algebraic correlations between Pauli operators, not spatial patterns. X/Y coherence operators (encoding entanglement) receive 90× mean attention.

### Scalability: Hierarchical Transformers (8-qubit)

Response to peer review Critique 1: Proof-of-concept for scaling beyond 5 qubits.

**Key Challenge**: Element-wise tokenization doesn't scale:
- 5 qubits: 1,024 tokens → O(1M) attention ops ✓
- 6 qubits: 4,096 tokens → O(16M) attention ops (expensive)
- 8 qubits: 65,536 tokens → O(4B) attention ops (impossible)

**Solution**: Hierarchical patch-based tokenization reduces tokens:
- Group elements into spatial patches
- Each patch → 1 token
- Example: 256×256 matrix → 32×32 patches → 64 tokens → O(4k) attention ops

#### Dataset Generation

| File | Description |
|------|-------------|
| `generate_8qubit_dataset.py` | Generates 8-qubit dataset (256×256 matrices, ~49GB, 100k samples) |
| `dataset_6qubit_backup/` | Backup of 6-qubit dataset (97 chunks, 6GB) - intermediate experiment |

**8-qubit Dataset Properties**:
- Matrix size: 256×256 (65,536 complex elements)
- Circuit depth: 6-9 layers (matching 5-qubit pattern)
- Noise types: depolarizing, amplitude damping, phase damping, bit flip, mixed
- Noise levels: 0.05, 0.10, 0.15, 0.20
- Total samples: 100,000 (100 chunks × 1000 samples)
- Storage: ~49GB

#### Hierarchical Models

| File | Description |
|------|-------------|
| `models/transformer_hierarchical_5qubit.py` | Hierarchical Transformer for 5-qubit (~1.1M params) |
| `models/transformer_hierarchical_8qubit.py` | Hierarchical Transformer for 8-qubit (~1.6M params) |
| `train_hierarchical_5qubit.py` | Training script with **timing logs** for 5-qubit hierarchical transformer |
| `train_hierarchical_8qubit.py` | Training script for 8-qubit with **streaming dataset** |
| `queue_jobs.sh` | Auto-run script: waits for 8-qubit generation → trains 5-qubit → trains 8-qubit |

**Hierarchical Transformer Architecture**:

| Component | 5-qubit | 8-qubit |
|-----------|---------|---------|
| Input | 32×32 | 256×256 |
| Patch size | 4×4 | 32×32 |
| Tokens | 64 | 64 |
| Embed dim | 128 | 128 |
| Layers | 4 enc + 4 dec | 4 enc + 4 dec |
| Parameters | 1,092,960 | 1,611,072 |

**Key insight**: Token count stays constant (64) regardless of matrix size. Only the patch embed/unembed layers grow with patch size.

#### Streaming Dataset (`training_loop/dataset/StreamingChunkDataset.py`)

For the 98GB 8-qubit dataset, loading all chunks into RAM causes OOM. The streaming dataset loads one chunk at a time:

```python
from training_loop.dataset.StreamingChunkDataset import (
    StreamingChunkDataset,
    get_chunk_files,
    split_chunk_files,
)

chunk_files = get_chunk_files("/workspace/dataset_8qubit")
train_files, val_files, test_files = split_chunk_files(chunk_files, seed=42)

# Streaming dataset for training (reshuffles each epoch)
train_dataset = StreamingChunkDataset(train_files, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=4)
```

**Purpose**: Compare element-wise vs hierarchical performance/speed to demonstrate scalability tradeoffs for reviewers.

#### Training Infrastructure (`train_14/`)

Experimental directory for 6-qubit/8-qubit hierarchical training (intermediate work, superseded by final 8-qubit approach).

| File | Description |
|------|-------------|
| `train_14/transformer_6qubit.py` | Hierarchical Transformer for 6-qubit (64×64 → 64 tokens) |
| `train_14/generate_6qubit_dataset.py` | 6-qubit dataset generator (intermediate) |
| `train_14/train.py` | Training script with disk-freeing for quota issues |
| `train_14/eval_uhlmann.py` | Uhlmann fidelity evaluation |

#### Monitoring Running Experiments

**Runpod Access**:
```bash
ssh -i ~/.ssh/id_ed25519 -p 20698 root@157.157.221.29
```

**GPU**: NVIDIA RTX A4000 (16GB)
**Storage**: 120GB allocated

##### Check Running Processes

```bash
# View all Python processes
ps aux | grep python

# Check specific processes
ps aux | grep generate_8qubit_dataset.py    # 8-qubit dataset generation
ps aux | grep train_hierarchical            # Hierarchical training
ps aux | grep queue_jobs.sh                 # Queue script
```

##### Monitor Dataset Generation Progress

```bash
# Count generated chunks
ls -1 /workspace/dataset_8qubit/*.pt | wc -l

# Check latest chunk timestamp (shows if still generating)
ls -lt /workspace/dataset_8qubit/*.pt | head -5

# Monitor in real-time (updates every 60 seconds)
watch -n 60 'ls -1 /workspace/dataset_8qubit/*.pt | wc -l'

# Check dataset size
du -sh /workspace/dataset_8qubit/

# View generation log (if running with nohup)
tail -f nohup.out
tail -f generate_8qubit.log  # If redirected to specific log file
```

##### Monitor Training Progress

```bash
# Check training logs
tail -f /workspace/csvs_2/hierarchical_transformer_5qubit.csv
tail -f /workspace/csvs_2/hierarchical_transformer_8qubit.csv

# Check timing logs (critical for paper)
tail -f /workspace/csvs_2/hierarchical_transformer_5qubit_timing.csv
tail -f /workspace/csvs_2/hierarchical_transformer_8qubit_timing.csv

# Check training process log output
tail -f /workspace/train_hierarchical_5qubit.log
tail -f /workspace/train_hierarchical_8qubit.log

# Count saved checkpoints
ls -1 /workspace/checkpoints_2/hierarchical_transformer_5qubit/ | wc -l
ls -1 /workspace/checkpoints_2/hierarchical_transformer_8qubit/ | wc -l

# Check most recent checkpoint
ls -lt /workspace/checkpoints_2/hierarchical_transformer_5qubit/ | head -5
```

##### Monitor System Resources

```bash
# Check GPU usage
nvidia-smi

# Detailed GPU monitoring (updates every 2 seconds)
watch -n 2 nvidia-smi

# Check disk usage
df -h /workspace

# Check detailed directory sizes
du -sh /workspace/*

# Check free space breakdown
du -h --max-depth=1 /workspace | sort -hr
```

##### Monitor Queue Script

```bash
# Check if queue script is running
ps aux | grep queue_jobs.sh

# View queue script log output
tail -f nohup.out | grep -A 5 "queue_jobs"

# Manually check queue status
bash /workspace/queue_jobs.sh --dry-run  # If implemented
```

##### Check 5-qubit Dataset Upload (from local machine)

```bash
# On local machine - check upload process
ps aux | grep "scp.*dataset_smaller"

# Count uploaded chunks (on runpod)
ls -1 /workspace/dataset_smaller/*.pt | wc -l

# Target: 1000 chunks total
# Check progress: (current_count / 1000) * 100%
```

##### Quick Status Check (All-in-One)

```bash
# SSH into runpod and run this comprehensive status check:
ssh -i ~/.ssh/id_ed25519 -p 20698 root@157.157.221.29 << 'EOF'
echo "=== GPU Status ==="
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader

echo -e "\n=== Disk Usage ==="
df -h /workspace | tail -1

echo -e "\n=== Running Processes ==="
ps aux | grep -E "(python|queue_jobs)" | grep -v grep

echo -e "\n=== Dataset Progress ==="
echo "8-qubit chunks: $(ls -1 /workspace/dataset_8qubit/*.pt 2>/dev/null | wc -l) / 100"
echo "5-qubit chunks: $(ls -1 /workspace/dataset_smaller/*.pt 2>/dev/null | wc -l) / 1000"

echo -e "\n=== Training Progress ==="
if [ -f /workspace/csvs_2/hierarchical_transformer_5qubit.csv ]; then
    echo "5-qubit training: epoch $(tail -1 /workspace/csvs_2/hierarchical_transformer_5qubit.csv | cut -d',' -f1)"
fi
if [ -f /workspace/csvs_2/hierarchical_transformer_8qubit.csv ]; then
    echo "8-qubit training: epoch $(tail -1 /workspace/csvs_2/hierarchical_transformer_8qubit.csv | cut -d',' -f1)"
fi

echo -e "\n=== Latest Logs (last 5 lines) ==="
if [ -f /workspace/train_hierarchical_5qubit.log ]; then
    echo "5-qubit training log:"
    tail -5 /workspace/train_hierarchical_5qubit.log
fi
EOF
```

##### Estimated Timeline

- **8-qubit dataset generation**: ~10-12 hours (100,000 samples at ~2.5 samples/sec)
- **5-qubit hierarchical training**: ~6-8 hours (100 epochs)
- **8-qubit hierarchical training**: ~12-15 hours (100 epochs, larger matrices)
- **Total pipeline**: ~30-36 hours from start

##### Troubleshooting

**Process died unexpectedly**:
```bash
# Check nohup.out for errors
tail -100 nohup.out

# Check for OOM kills
dmesg | grep -i "killed process"
```

**Disk full**:
```bash
# Find largest files
find /workspace -type f -size +1G -exec ls -lh {} \;

# Clean up old checkpoints (if needed)
rm /workspace/checkpoints_2/hierarchical_transformer_5qubit/epoch_[1-9].pt
rm /workspace/checkpoints_2/hierarchical_transformer_5qubit/epoch_[1-4][0-9].pt
```

**Port changed after restart**:
```bash
# Check runpod web UI for new port
# Remove old host key: ssh-keygen -R [157.157.221.29]:<old_port>
# Connect with new port: ssh -i ~/.ssh/id_ed25519 -p <new_port> root@157.157.221.29
```

---

## train_16: Float64 End-to-End Pipeline (CURRENT)

Complete float64 precision training pipeline for quantum density matrix denoising. Uses double precision throughout to ensure numerical stability during eigendecomposition for Uhlmann fidelity.

### Key Design Decisions

1. **All float64**: Models, data, and computations use `torch.float64`
2. **Hierarchical tokenization**: 64 tokens regardless of qubit count (4x4 patches for 5-qubit, 32x32 for 8-qubit)
3. **Parallel dataset generation**: ~10x speedup using multiprocessing (20 workers)
4. **Batch size = 256**: Max tested on RTX A4000 (16GB) with 8-qubit models using ~10GB VRAM
5. **VRAM logging**: For scalability arguments

### Dataset Generation (Parallel)

| File | Description |
|------|-------------|
| `train_16/generate_5qubit_float64_parallel.py` | Parallel 5-qubit dataset generator (~12 min for 100k samples) |
| `train_16/generate_8qubit_float64_parallel.py` | Parallel 8-qubit dataset generator (~1 hour for 100k samples) |
| `train_16/generate_5qubit_float64.py` | Sequential version (legacy, ~2 hours) |
| `train_16/generate_8qubit_float64.py` | Sequential version (legacy, ~10 hours) |

**Parallel speedup**: Uses `multiprocessing.Pool` with 20 workers (one per noise cell). Each cell generates 5000 samples independently with unique seed.

### Models (`train_16/models/`)

| File | Params | Description |
|------|--------|-------------|
| `transformer_5qubit.py` | ~1.09M | 5-qubit hierarchical Transformer (4+4 layers, 8 heads) |
| `transformer_8qubit.py` | ~1.61M | 8-qubit hierarchical Transformer (4+4 layers, 8 heads) |
| `mlp_5qubit.py` | ~1.09M | 5-qubit MLP-Mixer style (matched params) |
| `mlp_5qubit_wide.py` | ~5.2M | 5-qubit wide MLP (capacity control) |
| `mlp_5qubit_deep.py` | ~2.15M | 5-qubit deep MLP (8+8 layers, depth control) |
| `mlp_8qubit.py` | ~1.61M | 8-qubit MLP (matched params) |

### Training Scripts (`train_16/train/`)

| File | Description |
|------|-------------|
| `train_utils.py` | Shared training loop with VRAM logging, early stopping |
| `train_transformer_5qubit.py` | Train 5-qubit Transformer |
| `train_mlp_5qubit.py` | Train 5-qubit MLP (matched) |
| `train_mlp_5qubit_wide.py` | Train 5-qubit MLP (wide, ~5.2M params) |
| `train_mlp_5qubit_deep.py` | Train 5-qubit MLP (deep, 8+8 layers) |
| `train_transformer_8qubit.py` | Train 8-qubit Transformer |
| `train_mlp_8qubit.py` | Train 8-qubit MLP (matched) |

### Evaluation (`train_16/eval/`)

| File | Description |
|------|-------------|
| `eval_uhlmann.py` | Uhlmann fidelity evaluation with failure tracking and self-check |

### Utilities

| File | Description |
|------|-------------|
| `train_16/test_batch_size.py` | Binary search for max batch size on GPU (found 256 safe for 8-qubit) |

### Pipeline Script

| File | Description |
|------|-------------|
| `train_16/queue_float64.sh` | Complete pipeline: generate datasets → train all models → evaluate |

**Pipeline phases**:
1. Generate 5-qubit float64 dataset (parallel, ~12 min)
2. Train 5-qubit models (transformer, mlp, mlp_wide, mlp_deep) with Uhlmann eval after each
3. Generate 8-qubit float64 dataset (parallel, ~1 hour)
4. Train 8-qubit models (transformer, mlp) with Uhlmann eval after each

### Output Locations

| Directory | Description |
|-----------|-------------|
| `dataset_5qubit_float64/` | 5-qubit dataset (100 chunks, 100k samples) |
| `dataset_8qubit_float64/` | 8-qubit dataset (100 chunks, 100k samples) |
| `train_16/checkpoints_16/` | Model checkpoints (`best.pt` for each model) |
| `train_16/csvs_16/` | Training logs, VRAM logs, Uhlmann fidelity results |

### Monitoring Experiments

Use the status script for quick checks:

```bash
python .opencode/skill/check-experiments/scripts/status.py
```

This shows:
- Pipeline running status and current phase
- Progress percentage with ETA
- GPU/disk usage
- Dataset and checkpoint status
- Recent log output

---

## Miscellaneous Files

| File | Description |
|------|-------------|
| `emergency_notes.txt` | Quick notes about which csvs/checkpoints are canonical |
| `nohup.out` | Training logs from runpod (23MB, contains multiple runs) |
| `training_loop_3/` | Placeholder for paper v3 training loop (mostly empty) |
| `.opencode/skill/check-experiments/` | OpenCode skill for monitoring RunPod experiments |

---

## Dependencies

- PyTorch
- Cirq (for dataset generation)
- pandas, numpy, matplotlib
- tqdm

See `venv/` for the full virtual environment.