# AGENTS.md

**IMPORTANT: Make only a single edit per message. Do not batch multiple edits.**

Guidance for AI coding agents working on this repository.

## Project Overview

Research project comparing neural network architectures (CNN, MLP, Transformer) for quantum density matrix denoising. The goal is to recover clean quantum states from noisy measurements.

**Key finding**: Transformers significantly outperform CNNs and MLPs for this task, even when controlling for parameter count and removing spatial structure.

## Environment

### Local
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### RunPod (Remote)
```bash
# Always activate venv before running Python on RunPod
source /workspace/venv/bin/activate
python your_script.py
```

**RunPod SSH**: `ssh -i ~/.ssh/id_ed25519 -p <PORT> root@157.157.221.29`
- Port changes when volume storage is modified
- All training should be run on RunPod, not locally

- **Python**: 3.13.9
- **Key dependencies**: PyTorch, Cirq (quantum simulation), pandas, numpy, matplotlib, tqdm

## Quick Reference

### Common Commands

```bash
# Training
python train_models.py              # Train original CNN/Transformer models
python train_14/train.py            # Train hierarchical 6-qubit transformer (current focus)

# Evaluation
python eval_models_on_uhlmann.py    # Evaluate on Uhlmann fidelity (true quantum metric)
python eval_per_noise_cell.py       # Evaluate by noise type/level

# Figure generation
python figures/baseline_comparison_charts.py
```

### Data Format

- **Input/Output**: `(B, 2, 32, 32)` tensors for 5-qubit systems
  - Channel 0: Real part of 32x32 density matrix
  - Channel 1: Imaginary part
- **Complex conversion**: `rho = tensor[:, 0] + 1j * tensor[:, 1]`
- **Dataset location**: `dataset_smaller/` (~100k chunked .pt files)

## Directory Structure

### Active Code

| Directory | Purpose |
|-----------|---------|
| `models/` | Original architectures (CNN, Transformer, MLP) |
| `losses/` | Loss functions (Frobenius fidelity, physics constraints) |
| `training_loop/` | Reusable training infrastructure |
| `train_14/` | **Current focus**: 6-qubit hierarchical transformer |
| `train_11/` | Pauli representation experiments |
| `train_v8/` | Final MLP vs Transformer comparison |
| `figures/` | Figure generation scripts |

### Results & Checkpoints

| Directory | Contents |
|-----------|----------|
| `checkpoints_2/` | Paper v2 checkpoints (CNN/Transformer) |
| `csvs_2/` | Paper v2 results |
| `train_v8/checkpoints_8/` | Final v8 model checkpoints |
| `train_v8/csvs_8/` | Final v8 training logs |
| `train_11/checkpoints_11/` | Pauli model checkpoints |
| `train_14/checkpoints_14/` | 6-qubit hierarchical checkpoints |

### Legacy/Failed Experiments

| Directory | Status |
|-----------|--------|
| `models_3/` | Cholesky outputs - **FAILED** (collapsed to I/32) |
| `models_4/` | CLS global projection - **FAILED** |
| `models_5/` | Row-based tokenization - **FAILED** |
| `train_v6/` | Non-residual MLP - **FAILED** (destroyed info) |

## Key Conventions

### Reproducibility

**Always use `seed=42`** for train/val/test splits:

```python
from training_loop.dataset.split_chunks import split_chunks
train, val, test = split_chunks(chunks, 0.8, 0.1, seed=42)
```

### Model Interface

All models follow this pattern:

```python
class MyModel(nn.Module):
    def __init__(self, loss_fn=None):
        super().__init__()
        self.loss_fn = loss_fn
        # ... layers

    def forward(self, x):
        # x: (B, 2, 32, 32) -> (B, 2, 32, 32)
        pass

    def compute_loss(self, pred, target):
        return self.loss_fn(pred, target)
```

### Training Hyperparameters

| Model | Batch Size | LR | Optimizer | Epochs |
|-------|------------|-----|-----------|--------|
| CNN | 32 | 3e-4 | Adam | 100 |
| MLP | 64 | 3e-4 | AdamW | 100 |
| Transformer | 8 | 3e-4 | Adam | 100 |

Transformers use smaller batch size due to memory constraints.

### Loss Functions

```python
from losses.frob import FrobeniusFidelityLoss          # Primary loss
from losses.total_physics_loss import CompositePhysicsTotalLoss  # Frobenius + physics
```

## Metrics

### Uhlmann Fidelity (Ground Truth)

The true quantum fidelity metric: `F(ρ, σ) = (Tr[√(√ρ σ √ρ)])²`

- **Baseline** (noisy vs clean): ~0.11-0.12
- Used for all final evaluations, NOT for training

### Frobenius Fidelity (Training Proxy)

Cosine similarity for complex matrices. Used as training loss because it's differentiable and faster.

## Key Results

| Model | Params | Uhlmann Fidelity |
|-------|--------|------------------|
| Baseline (noisy) | — | 0.12 |
| MLP (residual) | ~117k | 0.17 (1.4x) |
| Transformer | ~119k | 0.28 (2.3x) |
| Wide MLP | ~1M | 0.19 |

**Conclusion**: Architecture matters more than capacity. Transformers learn pairwise correlations that MLPs cannot.

## Known Failure Modes

### Cholesky Output Layer

**DO NOT USE `models_3/`** - Outputs collapsed to maximally mixed state (I/32).

Root cause: Per-row output projection prevents global Cholesky coordination. See `debug_transformer_collapse/DIAGNOSIS.md`.

### Non-Residual MLP

**DO NOT USE `train_v6/mlp.py`** - Severe bottleneck destroyed information (fidelity worse than baseline).

**Fix**: Use residual architecture: `output = input + correction(input)` (see `train_v7/`, `train_v8/`).

## Adding New Experiments

1. Create new directory: `train_vN/` or `train_N/`
2. Include `__init__.py` with brief description
3. Follow existing patterns:
   - Import from `training_loop/` for data handling
   - Import from `losses/` for loss functions
   - Use `CSVLogger` for training logs
   - Save checkpoints to `checkpoints_N/`
   - Save logs to `csvs_N/`

4. Always evaluate on Uhlmann fidelity (not just Frobenius)
5. Compare to baseline (~0.12) and existing models

## Documentation

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Quick reference for main commands |
| `INDEX.md` | Comprehensive experiment documentation |
| `paper_v2.org` | CNN vs Transformer paper (Org-mode) |
| `paper_v3.org` | MLP vs Transformer paper (Org-mode) |

## Monitoring Remote Training

Training runs on RunPod pods. To check experiment status:

```bash
# Check training progress (grep for key lines)
ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no -p <PORT> root@<IP> \
  "grep -E '^Epoch|^Validation|Training complete|Early stopping' /workspace/<LOG_FILE>.log | tail -12"
```

### Current Pod Assignments (as of 2026-01-03)

| Model | IP | Port | Log File | Status |
|-------|-----|------|----------|--------|
| 5q MLP + Transformer | 157.157.221.29 | 20634 | train_17.log | COMPLETE |
| 5q MLP deep | 205.196.17.100 | 9718 | train_17_mlp_deep.log | COMPLETE (early stop epoch 89) |
| 5q MLP wide | 157.157.221.29 | 19758 | train_17_mlp_wide.log | COMPLETE (early stop epoch 60) |
| 8q Transformer | 157.157.221.29 | 19898 | train_17_transformer_8q.log | RUNNING (epoch ~55, val ~0.868) |
| 8q MLP | 157.157.221.29 | 20778 | train_17_mlp_8q.log | RUNNING (epoch ~72, val ~0.867) |

**Note**: Ports change when pods restart. Check RunPod dashboard for current ports.

### Example: Check all experiments at once

```bash
echo "--- 5q main ---" && \
ssh -i ~/.ssh/id_ed25519 -p 20634 root@157.157.221.29 \
  "grep -E '^Epoch|^Validation|Training complete|Early stopping' /workspace/train_17.log | tail -10"

echo "--- Transformer 8q ---" && \
ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no -p 19898 root@157.157.221.29 \
  "grep -E '^Epoch|^Validation|Training complete|Early stopping' /workspace/train_17_transformer_8q.log | tail -10"

echo "--- MLP 8q ---" && \
ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no -p 20778 root@157.157.221.29 \
  "grep -E '^Epoch|^Validation|Training complete|Early stopping' /workspace/train_17_mlp_8q.log | tail -10"
```

### Automated Monitoring

To monitor every 15 minutes:
```bash
sleep 900 && ssh ... "grep ..." 
```

Use `kdeconnect-cli --ring -n "Galaxy A16 5G"` to alert when training completes (requires KDE Connect).

## Tips

- Transformer needs `batch_size=8` (memory constraints)
- Check `csvs_2/uhlmann_ground_truth/` for baseline fidelity data
- Use `scripts/count_params.py` to count model parameters
- Models may early-stop before epoch 100; check actual training length in CSV logs
- `csvs_3/` is canonical for paper v3 results (ignore `csvs_3_old/`)
