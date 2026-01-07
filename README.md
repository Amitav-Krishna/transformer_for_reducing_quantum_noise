# Frobenius Normalization for Quantum State Denoising

This repository contains code and experiments for the paper "Frobenius Normalization Enables Stable Training for Quantum State Denoising".

## Key Finding

Frobenius normalization—normalizing density matrices to unit norm before training—is critical for stable neural network training on quantum state denoising. Without it, models struggle to learn both structural denoising and the ~15× scale correction caused by decoherence.

## Important Files

### Paper
- `paper_v4.org` — Main paper source (Org-mode)
- `paper_v4.pdf` — Compiled PDF
- `references.bib` — Bibliography

### Training Code (train_17/)
Primary experiments with Frobenius normalization:

| Script | Description |
|--------|-------------|
| `train_17/train/train.py` | 5-qubit Transformer and MLP training |
| `train_17/train/train_mlp_deep.py` | 5-qubit deep MLP ablation (8+8 layers) |
| `train_17/train/train_mlp_wide.py` | 5-qubit wide MLP ablation (4× hidden) |
| `train_17/train/train_transformer_8q.py` | 8-qubit Transformer training |
| `train_17/train/train_mlp_8q.py` | 8-qubit MLP training |

### Evaluation
| Script | Description |
|--------|-------------|
| `train_17/eval_uhlmann.py` | Evaluate 5-qubit models on Uhlmann fidelity |
| `train_17/eval_8q_uhlmann.py` | Evaluate 8-qubit models on Uhlmann fidelity |
| `train_17/eval_ablation.py` | Evaluate deep/wide MLP ablations |

### Results
Training logs and CSVs are in `train_17/results/`:
- `5q_transformer/` — 5-qubit Transformer (0.525 Uhlmann fidelity)
- `5q_mlp/` — 5-qubit MLP (0.516 Uhlmann fidelity)
- `5q_deep/` — 5-qubit deep MLP ablation (0.349 Uhlmann fidelity)
- `5q_wide/` — 5-qubit wide MLP ablation (0.344 Uhlmann fidelity)
- `8q_transformer/` — 8-qubit Transformer (0.019 Uhlmann fidelity)
- `8q_mlp/` — 8-qubit MLP (0.024 Uhlmann fidelity)

### Figures
- `figures/val_loss_curves.pdf` — Validation loss curves
- `figures/frob_vs_uhlmann_correlation.pdf` — Frobenius vs Uhlmann fidelity correlation

## Quick Start

```bash
# Setup
source venv/bin/activate
pip install -r requirements.txt

# Train 5-qubit models
python train_17/train/train.py

# Evaluate on Uhlmann fidelity
python train_17/eval_uhlmann.py
```

## Key Results

| Model | Params | Uhlmann Fidelity | vs Baseline |
|-------|--------|------------------|-------------|
| Baseline (noisy) | — | 0.167 | 1.00 times |
| MLP (5-qubit) | 1.09M | 0.516 | 3.09 times |
| Transformer (5-qubit) | 1.09M | 0.525 | 3.14 times |
| MLP (8-qubit) | 1.61M | 0.024 | 3.76 times |

With proper normalization, the Transformer-MLP gap narrows to ~1.7%, suggesting input conditioning matters more than architecture.

## Repository Structure

See `AGENTS.md` for detailed guidance on the codebase structure and `CLAUDE.md` for quick command reference.
