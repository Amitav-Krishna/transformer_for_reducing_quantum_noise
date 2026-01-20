#!/bin/bash
#
# Run all train_14 experiments (6-qubit scalability POC):
# 1. Generate 6-qubit dataset (100k samples)
# 2. Train Hierarchical Transformer
# 3. Evaluate on Uhlmann fidelity
#
# Usage:
#   cd /path/to/transformer_qnr
#   bash train_14/run_all.sh
#
# Or with nohup for long runs:
#   nohup bash train_14/run_all.sh > train_14/train_14.log 2>&1 &

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "Train_14: 6-Qubit Scalability Proof-of-Concept"
echo "Project directory: $PROJECT_DIR"
echo "=============================================="

cd "$PROJECT_DIR"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

echo ""
echo "=============================================="
echo "Step 1/3: Generating 6-qubit dataset"
echo "  - 100,000 samples (5 noise types × 4 levels × 5000)"
echo "  - 64×64 density matrices (4096 elements)"
echo "=============================================="
python train_14/generate_6qubit_dataset.py

echo ""
echo "=============================================="
echo "Step 2/3: Training Hierarchical Transformer"
echo "  - 64 patch tokens (8×8 patches)"
echo "  - ~500k parameters"
echo "=============================================="
python train_14/train.py

echo ""
echo "=============================================="
echo "Step 3/3: Evaluating on Uhlmann fidelity"
echo "=============================================="
python train_14/eval_uhlmann.py

echo ""
echo "=============================================="
echo "6-Qubit Scalability POC complete!"
echo ""
echo "Checkpoints: train_14/checkpoints_14/"
echo "Logs: train_14/csvs_14/"
echo "=============================================="
