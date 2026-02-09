# Reproducing Results

This directory contains the code, figures, and final PDF for the paper
*Frobenius Normalization Enables Stable Training for Quantum State Denoising*.

Repository: https://github.com/Amitav-Krishna/transformer_for_reducing_quantum_noise

## Prerequisites

- Python 3.12+
- CUDA-capable GPU (training was done on NVIDIA A100 / H100)
- ~4 GB disk for 5-qubit dataset, ~200 GB for 8-qubit dataset

```bash
pip install torch cirq numpy scipy pandas scikit-learn matplotlib
```

## Directory Structure

- `code/train_16/` — Dataset generation and model definitions
- `code/train_17/` — Main training and evaluation scripts (normalized + unnormalized)
- `code/train_19/` — Seed robustness ablation (seeds 100, 200)
- `code/training_loop/` — Shared evaluation utilities
- `figures/` — Generated figures used in the paper
- `paper_v5.pdf` — Final paper

## Step 1: Generate Datasets

### 5-qubit (float64)
```bash
python code/train_16/generate_5qubit_float64.py
```
Generates 100,000 samples (5 noise types x 4 noise levels x 5,000 samples each).
Output: chunked `.pt` files in `dataset_5qubit_float64/`.

### 8-qubit (float64)
```bash
python code/train_16/generate_8qubit_float64.py
```
Same structure but 256x256 density matrices. Output: `dataset_8qubit_float64/`.
This takes significantly longer and produces ~200 GB of data.

## Step 2: Train Models

All models use AdamW (lr=3e-4, weight_decay=1e-5), batch size 256, up to 100 epochs with early stopping (patience 15).

### 5-qubit models (normalized)
```bash
python code/train_17/train/train.py            # MLP + Transformer
```

### 5-qubit ablations (unnormalized)
```bash
python code/train_17/train/train_mlp_wide.py   # Wide MLP (5.29M params)
python code/train_17/train/train_mlp_deep.py   # Deep MLP (2.15M params)
```

### 8-qubit models (normalized)
```bash
python code/train_17/train/train_mlp_8q.py
python code/train_17/train/train_transformer_8q.py
```

Checkpoints are saved to `checkpoints_17/{model_name}/best.pt`.
Training logs (CSV) are saved alongside.

## Step 3: Evaluate

### Uhlmann fidelity on 5-qubit test set
```bash
python code/train_17/eval_uhlmann.py
```

### Uhlmann fidelity on 8-qubit test set
```bash
python code/train_17/eval_8q_uhlmann.py
```

### Capacity ablation evaluation
```bash
python code/train_17/eval_ablation.py
```

Results are written to `code/train_17/results/`.

## Step 4: Seed Robustness (Optional)

```bash
bash code/train_19/run_seeds.sh
```
Retrains 5-qubit models with seeds 100 and 200 (data split stays at seed=42).

## Expected Results

| Model                  | Qubit | Uhlmann Fidelity | vs Baseline |
|------------------------|-------|------------------|-------------|
| Baseline (noisy)       | 5     | 0.167            | 1.00x       |
| MLP (normalized)       | 5     | 0.516            | 3.09x       |
| Transformer (norm.)    | 5     | 0.525            | 3.14x       |
| MLP (unnormalized)     | 5     | 0.463            | 2.77x       |
| MLP wide (unnorm.)     | 5     | 0.310            | 1.86x       |
| MLP deep (unnorm.)     | 5     | 0.407            | 2.44x       |
| Baseline (noisy)       | 8     | 0.006            | 1.00x       |
| MLP (normalized)       | 8     | 0.024            | 3.76x       |
| Transformer (norm.)    | 8     | 0.019            | 3.00x       |

## Notes

- All datasets use seed=42 for reproducible 80/10/10 train/val/test splits, stratified by noise cell.
- Double precision (float64) is required for numerically stable Uhlmann fidelity computation.
- The 8-qubit dataset is streamed from disk due to its size (~200 GB).
