#!/bin/bash
# =============================================================================
# Float64 Benchmark Queue Script for train_16 (Parallel Version)
# =============================================================================
# 
# This script runs the complete float64 training pipeline:
# 1. Generate 5-qubit float64 dataset (PARALLEL - ~15 min)
# 2. Train all 5-qubit models with evaluation after each
# 3. Generate 8-qubit float64 dataset (PARALLEL - ~1 hour)
# 4. Train all 8-qubit models with evaluation after each
#
# Usage:
#   cd /workspace
#   source venv/bin/activate
#   bash train_16/queue_float64.sh 2>&1 | tee train_16_run.log
#
# =============================================================================

set -e  # Exit on error

# Timestamp function
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

echo "=============================================="
echo "Float64 Benchmark Pipeline (Parallel)"
echo "Started: $(timestamp)"
echo "=============================================="

# =============================================================================
# Phase 1: 5-Qubit Dataset Generation (PARALLEL)
# =============================================================================
echo ""
echo "=============================================="
echo "Phase 1: Generating 5-qubit float64 dataset (PARALLEL)"
echo "Started: $(timestamp)"
echo "=============================================="

if [ -d "dataset_5qubit_float64" ] && [ "$(ls -A dataset_5qubit_float64 2>/dev/null)" ]; then
    echo "Dataset already exists, skipping generation"
else
    python train_16/generate_5qubit_float64_parallel.py
fi

echo "5-qubit dataset complete: $(timestamp)"

# =============================================================================
# Phase 2: 5-Qubit Model Training
# =============================================================================
echo ""
echo "=============================================="
echo "Phase 2: Training 5-qubit models"
echo "=============================================="

# 2a: Transformer 5-qubit
echo ""
echo "----------------------------------------------"
echo "Training: transformer_5qubit"
echo "Started: $(timestamp)"
echo "----------------------------------------------"
python -m train_16.train.train_transformer_5qubit

echo ""
echo "Evaluating: transformer_5qubit"
python train_16/eval/eval_uhlmann.py \
    --model transformer_5qubit \
    --checkpoint train_16/checkpoints_16/transformer_5qubit/best.pt \
    --dataset dataset_5qubit_float64

echo "transformer_5qubit complete: $(timestamp)"

# 2b: MLP 5-qubit (matched)
echo ""
echo "----------------------------------------------"
echo "Training: mlp_5qubit"
echo "Started: $(timestamp)"
echo "----------------------------------------------"
python -m train_16.train.train_mlp_5qubit

echo ""
echo "Evaluating: mlp_5qubit"
python train_16/eval/eval_uhlmann.py \
    --model mlp_5qubit \
    --checkpoint train_16/checkpoints_16/mlp_5qubit/best.pt \
    --dataset dataset_5qubit_float64

echo "mlp_5qubit complete: $(timestamp)"

# 2c: MLP 5-qubit wide (capacity control)
echo ""
echo "----------------------------------------------"
echo "Training: mlp_5qubit_wide"
echo "Started: $(timestamp)"
echo "----------------------------------------------"
python -m train_16.train.train_mlp_5qubit_wide

echo ""
echo "Evaluating: mlp_5qubit_wide"
python train_16/eval/eval_uhlmann.py \
    --model mlp_5qubit_wide \
    --checkpoint train_16/checkpoints_16/mlp_5qubit_wide/best.pt \
    --dataset dataset_5qubit_float64

echo "mlp_5qubit_wide complete: $(timestamp)"

# 2d: MLP 5-qubit deep (depth control)
echo ""
echo "----------------------------------------------"
echo "Training: mlp_5qubit_deep"
echo "Started: $(timestamp)"
echo "----------------------------------------------"
python -m train_16.train.train_mlp_5qubit_deep

echo ""
echo "Evaluating: mlp_5qubit_deep"
python train_16/eval/eval_uhlmann.py \
    --model mlp_5qubit_deep \
    --checkpoint train_16/checkpoints_16/mlp_5qubit_deep/best.pt \
    --dataset dataset_5qubit_float64

echo "mlp_5qubit_deep complete: $(timestamp)"

echo ""
echo "=============================================="
echo "Phase 2 complete: All 5-qubit models trained"
echo "Finished: $(timestamp)"
echo "=============================================="

# =============================================================================
# Phase 3: 8-Qubit Dataset Generation (PARALLEL)
# =============================================================================
echo ""
echo "=============================================="
echo "Phase 3: Generating 8-qubit float64 dataset (PARALLEL)"
echo "Started: $(timestamp)"
echo "=============================================="

if [ -d "dataset_8qubit_float64" ] && [ "$(ls -A dataset_8qubit_float64 2>/dev/null)" ]; then
    echo "Dataset already exists, skipping generation"
else
    python train_16/generate_8qubit_float64_parallel.py
fi

echo "8-qubit dataset complete: $(timestamp)"

# =============================================================================
# Phase 4: 8-Qubit Model Training
# =============================================================================
echo ""
echo "=============================================="
echo "Phase 4: Training 8-qubit models"
echo "=============================================="

# 4a: Transformer 8-qubit
echo ""
echo "----------------------------------------------"
echo "Training: transformer_8qubit"
echo "Started: $(timestamp)"
echo "----------------------------------------------"
python -m train_16.train.train_transformer_8qubit

echo ""
echo "Evaluating: transformer_8qubit"
python train_16/eval/eval_uhlmann.py \
    --model transformer_8qubit \
    --checkpoint train_16/checkpoints_16/transformer_8qubit/best.pt \
    --dataset dataset_8qubit_float64

echo "transformer_8qubit complete: $(timestamp)"

# 4b: MLP 8-qubit (matched)
echo ""
echo "----------------------------------------------"
echo "Training: mlp_8qubit"
echo "Started: $(timestamp)"
echo "----------------------------------------------"
python -m train_16.train.train_mlp_8qubit

echo ""
echo "Evaluating: mlp_8qubit"
python train_16/eval/eval_uhlmann.py \
    --model mlp_8qubit \
    --checkpoint train_16/checkpoints_16/mlp_8qubit/best.pt \
    --dataset dataset_8qubit_float64

echo "mlp_8qubit complete: $(timestamp)"

echo ""
echo "=============================================="
echo "Phase 4 complete: All 8-qubit models trained"
echo "Finished: $(timestamp)"
echo "=============================================="

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "PIPELINE COMPLETE"
echo "Finished: $(timestamp)"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  - train_16/checkpoints_16/  (model checkpoints)"
echo "  - train_16/csvs_16/         (training logs)"
echo ""
echo "Models trained:"
echo "  5-qubit: transformer_5qubit, mlp_5qubit, mlp_5qubit_wide, mlp_5qubit_deep"
echo "  8-qubit: transformer_8qubit, mlp_8qubit"
echo ""
