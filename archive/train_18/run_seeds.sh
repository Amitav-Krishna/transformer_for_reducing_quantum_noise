#!/bin/bash
#
# Seed ablation experiments for 5-qubit models.
#
# Runs Transformer and MLP with seeds 100 and 200 (original experiments used seed 42).
# Order: Transformer -> MLP for each seed (sequential execution).
#
# Usage (local):
#   cd /path/to/transformer_qnr
#   source venv/bin/activate
#   bash train_18/run_seeds.sh
#
# Usage (RunPod):
#   source /workspace/venv/bin/activate
#   bash train_18/run_seeds.sh 2>&1 | tee train_18_seeds.log

set -e  # Exit on error

echo "============================================================"
echo "Seed ablation experiments for 5-qubit models"
echo "Seeds: 100, 200 (original: 42)"
echo "Order: transformer -> mlp (for each seed)"
echo "============================================================"
echo ""

# Seed 100: Transformer
echo "[1/4] Training Transformer with seed 100..."
python train_18/train.py --seed 100 --model transformer
echo ""

# Seed 100: MLP
echo "[2/4] Training MLP with seed 100..."
python train_18/train.py --seed 100 --model mlp
echo ""

# Seed 200: Transformer
echo "[3/4] Training Transformer with seed 200..."
python train_18/train.py --seed 200 --model transformer
echo ""

# Seed 200: MLP
echo "[4/4] Training MLP with seed 200..."
python train_18/train.py --seed 200 --model mlp
echo ""

echo "============================================================"
echo "All seed ablation experiments complete!"
echo ""
echo "Checkpoints:"
echo "  train_18/checkpoints_18_seed100/transformer/"
echo "  train_18/checkpoints_18_seed100/mlp/"
echo "  train_18/checkpoints_18_seed200/transformer/"
echo "  train_18/checkpoints_18_seed200/mlp/"
echo ""
echo "Training logs:"
echo "  train_18/csvs_18_seed100/"
echo "  train_18/csvs_18_seed200/"
echo "============================================================"
