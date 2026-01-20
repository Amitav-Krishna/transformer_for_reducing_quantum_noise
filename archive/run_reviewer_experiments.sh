#!/bin/bash
#
# Run all reviewer-requested experiments:
# 1. Wait for train_13 (element-wise Cholesky) to finish
# 2. Evaluate train_13 on Uhlmann fidelity
# 3. Generate 6-qubit dataset (100k samples)
# 4. Train Hierarchical Transformer on 6-qubit data
# 5. Evaluate on Uhlmann fidelity
#
# Usage:
#   cd /workspace
#   source venv/bin/activate
#   nohup bash run_reviewer_experiments.sh > reviewer_experiments.log 2>&1 &

set -e

echo "=============================================="
echo "Running Reviewer-Requested Experiments"
echo "Started at: $(date)"
echo "=============================================="

cd /workspace

# Step 1: Wait for train_13 to finish (check every 30 seconds)
echo ""
echo "Step 1: Waiting for train_13 (element-wise Cholesky) to complete..."
while pgrep -f "train_13/cholesky/train.py" > /dev/null; do
    echo "  $(date): train_13 still running..."
    sleep 30
done
echo "  train_13 completed at $(date)"

# Step 2: Evaluate train_13
echo ""
echo "=============================================="
echo "Step 2: Evaluating train_13 (element-wise Cholesky)"
echo "=============================================="
python train_13/eval_uhlmann.py

# Step 3: Generate 6-qubit dataset
echo ""
echo "=============================================="
echo "Step 3: Generating 6-qubit dataset (100k samples)"
echo "=============================================="
python train_14/generate_6qubit_dataset.py

# Step 4: Train Hierarchical Transformer
echo ""
echo "=============================================="
echo "Step 4: Training Hierarchical Transformer (6-qubit)"
echo "=============================================="
python train_14/train.py

# Step 5: Evaluate on Uhlmann fidelity
echo ""
echo "=============================================="
echo "Step 5: Evaluating Hierarchical Transformer"
echo "=============================================="
python train_14/eval_uhlmann.py

echo ""
echo "=============================================="
echo "All Reviewer Experiments Complete!"
echo "Finished at: $(date)"
echo ""
echo "Results:"
echo "  train_13 (element-wise Cholesky): checkpoints_13/"
echo "  train_14 (6-qubit hierarchical): checkpoints_14/"
echo "=============================================="
