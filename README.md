# Comparing Machine Learning Strategies for Quantum Noise Reduction

This project compares CNN and Transformer autoencoders for density-matrix denoising across multiple quantum noise channels.

## Project Structure

```
.
├── paper_v2.org            # Main paper (Org-mode source)
├── paper_v2.pdf            # Compiled paper
├── references.bib          # Bibliography
│
├── dataset_smaller/        # Chunked training dataset (100k samples)
├── dataset.pt              # Original full dataset (1M samples)
│
├── models/                 # Model architectures
├── losses/                 # Loss functions
├── training_loop/          # Training infrastructure
│
├── checkpoints_2/          # Current model checkpoints (used in paper)
├── csvs_2/                 # Current experiment results (used in paper)
├── figures/                # Generated figures for paper
│
├── train_models.py         # Main training script
├── generate_dataset.py     # Dataset generation script
├── ground_truth_fidelity.py # Baseline fidelity computation
├── eval_models_on_uhlmann.py # Uhlmann fidelity evaluation
├── eval_per_noise_cell.py  # Per-noise-type/level evaluation
└── eval_models.py          # General model evaluation
```

---

## Scripts

### Training & Evaluation

| Script | Description |
|--------|-------------|
| `train_models.py` | Main entry point. Trains all 4 model configurations (CNN/Transformer x Frobenius/Physics loss) |
| `eval_models_on_uhlmann.py` | Evaluates trained models using Uhlmann fidelity on test set |
| `eval_per_noise_cell.py` | Evaluates models broken down by noise type and noise level |
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
| `cnn.py` | CNN Autoencoder - treats density matrix as 2-channel (real/imag) image |
| `transformer.py` | Transformer Autoencoder - treats density matrix as 1024 tokens |

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

### Model Checkpoints

| Path | Description |
|------|-------------|
| `checkpoints_2/cnn_frob/` | CNN + Frobenius loss (100 epochs + `best.pt`) |
| `checkpoints_2/cnn_physics/` | CNN + Physics loss (100 epochs + `best.pt`) |
| `checkpoints_2/transformer_frob/` | Transformer + Frobenius loss (100 epochs + `best.pt`) |
| `checkpoints_2/transformer_physics/` | Transformer + Physics loss (100 epochs + `best.pt`) |
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

---

## Usage

### Training

```bash
# Activate virtual environment
source venv/bin/activate

# Train all models (modify train_models.py to select which to run)
python train_models.py
```

### Evaluation

```bash
# Compute baseline fidelity (noisy vs clean)
python ground_truth_fidelity.py

# Evaluate models on Uhlmann fidelity
python eval_models_on_uhlmann.py

# Evaluate per noise type/level
python eval_per_noise_cell.py
```

### Generating Figures

Figures are generated via code blocks in `paper_v2.org` or standalone scripts in `figures/`:

```bash
python figures/baseline_comparison_charts.py
python figures/transformer_architecture.py
```

---

## Key Results

| Model | Uhlmann Fidelity | Improvement over Baseline |
|-------|------------------|---------------------------|
| Baseline (noisy input) | 0.11 ± 0.15 | — |
| CNN Frobenius | 0.30 ± 0.28 | +0.19 |
| CNN Physics | 0.06 ± 0.05 | -0.05 (worse than baseline) |
| Transformer Frobenius | 0.95 ± 0.12 | +0.84 |
| Transformer Physics | 0.33 ± 0.33 | +0.22 |

---

## Dependencies

- PyTorch
- Cirq (for dataset generation)
- pandas, numpy, matplotlib
- tqdm

See `venv/` for the full virtual environment.