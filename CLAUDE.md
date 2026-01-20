# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project comparing CNN and Transformer autoencoders for quantum density matrix denoising. The paper is in `paper_v2.org` (Org-mode source).

## Commands

```bash
# Setup
source venv/bin/activate
pip install -r requirements.txt

# Training
python train_models.py                      # Train all 4 original models
python train_transformer_matched_params.py  # Train capacity-matched transformer (~750k params)

# Evaluation
python ground_truth_fidelity.py    # Compute baseline (noisy vs clean) Uhlmann fidelity
python eval_models_on_uhlmann.py   # Evaluate all models on Uhlmann fidelity
python eval_per_noise_cell.py      # Evaluate by noise type/level

# Figures
python figures/baseline_comparison_charts.py
python figures/uhlmann_vs_frobenius_baseline.py
```

## Architecture

### Data Flow
1. `generate_dataset.py` creates density matrices with Cirq → `dataset_smaller/` (chunked .pt files)
2. `training_loop/dataset/load_chunks.py` loads chunks with metadata (noise_type, noise_level)
3. `training_loop/dataset/split_chunks.py` splits 80/10/10 train/val/test (seed=42)
4. Models in `models/` process input shape `(B, 2, 32, 32)` where channels are real/imag parts

### Models
- **CNN** (`models/cnn.py`): 748,898 params. Channels: 2→48→96→192→bottleneck→decode
- **Transformer** (`models/transformer.py`): 119,506 params. Flattens to 1024 tokens, embed_dim=32
- **TransformerMatched** (`models/transformer_matched_params.py`): 751,186 params. embed_dim=96, ffn_dim=128

### Losses (`losses/`)
- `frob.py`: Normalized Frobenius fidelity (cosine similarity for complex matrices)
- `physics.py`: Hermiticity + unit trace + positive semi-definiteness penalties
- `total_physics_loss.py`: Frobenius + 0.1 * physics

### Key Metrics
- **Uhlmann fidelity**: True quantum fidelity F(ρ,σ) = (Tr[√(√ρ σ √ρ)])²
- **Frobenius fidelity**: Proxy metric used for training loss

## Data Locations

- **Checkpoints**: `checkpoints_2/{model_name}/best.pt`
- **Training logs**: `csvs_2/{model_name}.csv`
- **Per-noise-cell results**: `csvs_2/noise_cells/{model_name}_noise_cells.csv`
- **Baseline fidelity**: `csvs_2/uhlmann_ground_truth/noisy_vs_clean_test_uhlmann.csv`

## Important Notes

- Dataset uses seed=42 for reproducible train/val/test splits
- Batch size: 32 for CNN, 8 for Transformer (due to memory)
- All models train for 100 epochs with Adam lr=3e-4
- The 0.11 baseline fidelity comes from `ground_truth_fidelity.py`