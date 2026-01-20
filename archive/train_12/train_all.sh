#!/bin/bash
#
# Train all models in train_12:
# 1. Cholesky-parameterized models (MLP + Transformer)
# 2. CNN baseline
#
# Usage:
#   cd /path/to/transformer_qnr
#   bash train_12/train_all.sh
#
# Or with nohup for long runs:
#   nohup bash train_12/train_all.sh > train_12/train_all.log 2>&1 &

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "Training all train_12 models"
echo "Project directory: $PROJECT_DIR"
echo "=========================================="

cd "$PROJECT_DIR"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

echo ""
echo "=========================================="
echo "Step 1/2: Training Cholesky models"
echo "  - MLP Cholesky (~117k params)"
echo "  - Transformer Cholesky (~119k params)"
echo "=========================================="
python train_12/cholesky/train.py

echo ""
echo "=========================================="
echo "Step 2/2: Training CNN baseline"
echo "  - CNN (~118k params)"
echo "=========================================="
python train_12/cnn/train.py

echo ""
echo "=========================================="
echo "All training complete!"
echo ""
echo "Checkpoints saved to:"
echo "  - train_12/checkpoints_12/cholesky/mlp_cholesky/"
echo "  - train_12/checkpoints_12/cholesky/transformer_cholesky/"
echo "  - train_12/checkpoints_12/cnn/"
echo ""
echo "Training logs saved to:"
echo "  - train_12/csvs_12/"
echo "=========================================="
