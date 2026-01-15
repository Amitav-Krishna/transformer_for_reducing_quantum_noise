#!/bin/bash
# Run seed ablation experiments with fixed data split (seed=42)
# Model initialization varies across seeds 100, 200

# Activate venv (adjust path as needed)
# source /workspace/venv/bin/activate

echo "=== Seed Ablation (v19) - Fixed data split, varying model init ==="

# Seed 100
echo "Training transformer with init seed 100..."
python train_19/train.py --seed 100 --model transformer

echo "Training MLP with init seed 100..."
python train_19/train.py --seed 100 --model mlp

# Seed 200
echo "Training transformer with init seed 200..."
python train_19/train.py --seed 200 --model transformer

echo "Training MLP with init seed 200..."
python train_19/train.py --seed 200 --model mlp

echo "=== All seed ablation experiments complete ==="
