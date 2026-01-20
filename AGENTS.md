# AGENTS.md

**IMPORTANT: Make only a single edit per message. Do not batch multiple edits.**

Guidance for AI coding agents working on this repository.

## Project Overview

Research project investigating the impact of **Frobenius normalization** on neural network training for quantum density matrix denoising.

**Key finding**: Frobenius normalization (normalizing inputs/targets to unit norm) is critical for training, enabling stable convergence and significantly improving performance (+25% Uhlmann fidelity for Transformers). It resolves the scale mismatch caused by decoherence. With normalization, architectural differences become less critical, though Transformers still show a stronger benefit (+25% vs +11%).

## Environment

### Local
```bash
# Setup
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

## Directory Structure

The repository has been reorganized (Jan 2026) to separate the current paper submission from historical experiments.

| Directory | Purpose |
|-----------|---------|
| `current/` | **ACTIVE**: Contains all code, figures, and text for the current paper submission (Paper v4). |
| `current/code/` | Active training code (`train_17`, `train_19`, `training_loop`). |
| `current/figures/` | Figures used in the paper. |
| `archive/` | Historical experiments, old papers, and legacy code (`train_11`-`15`, `train_v*`). |

### Key Code Directories (inside `current/code/`)

| Directory | Purpose |
|-----------|---------|
| `train_17/` | **Main Results**: Normalized training for MLP/Transformer (5q & 8q). |
| `train_16/` | **Ablation**: Unnormalized training (to demonstrate failure/poor performance). |
| `train_19/` | **Seed Robustness**: Seed ablation experiments with fixed data split. |
| `training_loop/` | Shared infrastructure (Dataset, generic training loop). |

## Quick Reference

### Common Commands (from `current/code/`)

```bash
# Training (Main Normalized Models)
cd current/code
python train_17/train.py

# Seed Ablation
python train_19/train.py --seed 100  # Uses fixed data split (seed 42)

# Evaluation
# Note: Evals are typically run on checkpoints generated in train_*/checkpoints_*/
python train_17/eval/eval_uhlmann.py ...
```

### Data Format

- **Input/Output**: `(B, 2, 32, 32)` tensors for 5-qubit systems ($2^5 \times 2^5$)
  - Channel 0: Real part
  - Channel 1: Imaginary part
- **Dataset**: `dataset_smaller/` (~100k chunked .pt files)
- **Normalization**: Input matrices have Frobenius norm $\ll 1$ (due to noise). Target matrices have Frobenius norm $= 1$ (pure states).

## Key Results (Paper v4)

| Metric | Baseline (Noisy) | Unnormalized (MLP) | Normalized (MLP) | Normalized (Transf.) |
|--------|------------------|--------------------|------------------|----------------------|
| **Uhlmann Fidelity** | 0.167 | 0.463 | 0.516 (+11%) | 0.525 (+25%) |

**Conclusion**: Normalization allows models to focus on structural denoising rather than scale correction. It is essential for scaling to 8-qubit systems.

## Monitoring Remote Training

Training runs on RunPod pods. To check experiment status:

```bash
# Check training progress (grep for key lines)
ssh -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no -p <PORT> root@<IP> \
  "grep -E '^Epoch|^Validation|Training complete|Early stopping' /workspace/<LOG_FILE>.log | tail -12"
```

### Active Experiments (Jan 2026)

| Model | Port | Log File | Status |
|-------|------|----------|--------|
| **8q Transformer** | 19898 | `train_17_transformer_8q.log` | RUNNING |
| **8q MLP** | 20778 | `train_17_mlp_8q.log` | RUNNING |

(Note: Local code is in `current/code/`, but remote pods may still use flat structure `/workspace/train_17/`).

## Key Conventions

### Reproducibility

- **Data Split**: Always use `seed=42` for train/val/test splitting.
- **Model Init**: Vary seed for initialization robustness (as in `train_19`).

### Paper Versioning

- `current/paper_v4.org` is the source of truth.
- `current/paper_v4.tex` is generated from it.
- Do not edit `.tex` directly if possible; edit `.org` and export.

## Known Issues

- **Unnormalized Training**: Fails to converge or plateaus early, especially on larger systems (8-qubit).
- **Capacity Ablation**: Increasing model size (width/depth) *hurts* performance without normalization due to optimization difficulties.
