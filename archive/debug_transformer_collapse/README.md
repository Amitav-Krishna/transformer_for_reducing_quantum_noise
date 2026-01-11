# Transformer Cholesky Model Collapse - Debugging Analysis

## Summary

The TransformerCholeskyAutoencoder has collapsed to outputting the maximally mixed state (I/32), achieving ~0.031 fidelity regardless of input. This folder contains reproducible scripts to diagnose the issue.

## Key Findings

1. **Training never started**: Val loss 0.8233 → 0.8237 over 24 epochs (flat)
2. **Outputs maximally mixed state**: ~0.031 ≈ 1/32 fidelity
3. **Critical architectural flaw**: `decoder(enc, enc)` creates degenerate autoencoder
4. **No global mixing**: Row-wise output projection prevents Cholesky coordination
5. **Discarded CLS token**: Global aggregation token thrown away before output

## Debugging Scripts

Run in order:

1. `verify_maximally_mixed.py` - Confirms output is I/32
2. `analyze_training_curves.py` - Shows training never improved
3. `test_decoder_architecture.py` - Demonstrates decoder redundancy
4. `test_output_mixing.py` - Shows lack of global coordination
5. `gradient_flow_analysis.py` - Checks for gradient issues
6. `compare_with_mlp.py` - Highlights why MLP succeeds

## Root Cause

The transformer uses `self.decoder(enc, enc)` which:
- Does self-attention on encoder output
- Then cross-attends to the same encoder output
- Creates redundant computation with no bottleneck
- Cannot learn meaningful representations

Combined with row-wise output projection that prevents global mixing needed for Cholesky factorization.

## Files

- Model: `/home/work/codage/transformer_qnr/models_3/transformer_cholesky.py`
- Training: `/home/work/codage/transformer_qnr/train_cholesky.py`
- Checkpoint: `/home/work/codage/transformer_qnr/checkpoints_3/transformer_cholesky/best.pt`
- Results: `/home/work/codage/transformer_qnr/csvs_3/noise_cells/transformer_cholesky_noise_cells.csv`
