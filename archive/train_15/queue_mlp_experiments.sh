#!/bin/bash
# Queue script to run hierarchical MLP experiments after Transformer finishes
#
# Usage: nohup bash queue_mlp_experiments.sh > queue_mlp.log 2>&1 &

set -e

cd /workspace
source /workspace/venv/bin/activate

echo "=== Hierarchical MLP Training Queue ==="
echo "Started at: $(date)"

# Wait for 8-qubit Transformer to finish
echo ""
echo "Waiting for 8-qubit Transformer training to complete..."
while pgrep -f "train_hierarchical_8qubit.py" > /dev/null; do
    EPOCHS=$(wc -l < /workspace/csvs_2/hierarchical_transformer_8qubit_val.csv 2>/dev/null || echo "1")
    EPOCHS=$((EPOCHS - 1))  # Subtract header
    echo "  $(date +%H:%M:%S) - Transformer at epoch $EPOCHS/100"
    sleep 300  # Check every 5 minutes
done

echo ""
echo "8-qubit Transformer training complete!"
echo "Final epochs: $(wc -l < /workspace/csvs_2/hierarchical_transformer_8qubit_val.csv)"

# Run 5-qubit MLP (matched params, ~1.5 hours)
echo ""
echo "=== Starting 5-qubit Hierarchical MLP (matched) Training ==="
echo "Started at: $(date)"
python -u /workspace/train_15/train_5qubit.py
echo "5-qubit MLP (matched) complete at: $(date)"

# Run 5-qubit Wide MLP (capacity control, ~2 hours)
echo ""
echo "=== Starting 5-qubit Hierarchical MLP Wide (~5.2M params) Training ==="
echo "Started at: $(date)"
python -u /workspace/train_15/train_5qubit_wide.py
echo "5-qubit MLP Wide complete at: $(date)"

# Run 8-qubit MLP (slower, ~40 hours)
echo ""
echo "=== Starting 8-qubit Hierarchical MLP Training ==="
echo "Started at: $(date)"
python -u /workspace/train_15/train.py
echo "8-qubit MLP complete at: $(date)"

echo ""
echo "=== All MLP experiments complete ==="
echo "Finished at: $(date)"
