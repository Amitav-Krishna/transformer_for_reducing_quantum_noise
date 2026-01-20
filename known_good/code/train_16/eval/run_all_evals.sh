#!/bin/bash
# Evaluate all train_16 (unnormalized) 5-qubit models on Uhlmann fidelity

source /workspace/venv/bin/activate
cd /workspace

echo "=============================================="
echo "Evaluating train_16 (UNNORMALIZED) checkpoints"
echo "=============================================="
echo ""

echo "=== MLP 5-qubit (matched) ==="
python train_16/eval/eval_uhlmann.py \
    --model mlp_5qubit \
    --checkpoint /workspace/train_16/checkpoints_16/mlp_5qubit/best.pt \
    --dataset dataset_5qubit_float64

echo ""
echo "=== MLP 5-qubit (wide) ==="
python train_16/eval/eval_uhlmann.py \
    --model mlp_5qubit_wide \
    --checkpoint /workspace/train_16/checkpoints_16/mlp_5qubit_wide/best.pt \
    --dataset dataset_5qubit_float64

echo ""
echo "=== MLP 5-qubit (deep) ==="
python train_16/eval/eval_uhlmann.py \
    --model mlp_5qubit_deep \
    --checkpoint /workspace/train_16/checkpoints_16/mlp_5qubit_deep/best.pt \
    --dataset dataset_5qubit_float64

echo ""
echo "=== Transformer 5-qubit ==="
python train_16/eval/eval_uhlmann.py \
    --model transformer_5qubit \
    --checkpoint /workspace/train_16/checkpoints_16/transformer_5qubit/best.pt \
    --dataset dataset_5qubit_float64

echo ""
echo "=============================================="
echo "All evaluations complete!"
echo "=============================================="
