#!/bin/bash
# Run all diagnostic scripts in sequence
# This reproduces the complete analysis of the Transformer Cholesky collapse

echo "=========================================================================="
echo "TRANSFORMER CHOLESKY MODEL COLLAPSE - COMPLETE DIAGNOSTICS"
echo "=========================================================================="
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "1/6: Verifying model outputs maximally mixed state..."
echo "----------------------------------------------------------------------"
python3 verify_maximally_mixed.py
echo ""

echo "2/6: Analyzing training curves..."
echo "----------------------------------------------------------------------"
python3 analyze_training_curves.py
echo ""

echo "3/6: Testing decoder architecture flaw..."
echo "----------------------------------------------------------------------"
python3 test_decoder_architecture.py
echo ""

echo "4/6: Testing output mixing problem..."
echo "----------------------------------------------------------------------"
python3 test_output_mixing.py
echo ""

echo "5/6: Analyzing gradient flow issues..."
echo "----------------------------------------------------------------------"
python3 gradient_flow_analysis.py
echo ""

echo "6/6: Comparing with MLP architecture..."
echo "----------------------------------------------------------------------"
python3 compare_with_mlp.py
echo ""

echo "=========================================================================="
echo "DIAGNOSTICS COMPLETE"
echo "=========================================================================="
echo ""
echo "Summary of findings:"
echo "  1. Model outputs I/32 (maximally mixed state) for all inputs"
echo "  2. Training never improved (val loss 0.823 → 0.823)"
echo "  3. decoder(enc, enc) creates degenerate autoencoder"
echo "  4. Per-row output projection prevents global coordination"
echo "  5. CLS token discarded, preventing global mixing"
echo "  6. MLP succeeds due to forced bottleneck + global mixing"
echo ""
echo "See DIAGNOSIS.md for complete analysis and recommendations"
echo ""
