# Transformer Cholesky Collapse - Analysis Index

## Quick Start

1. **Read first:** `QUICK_SUMMARY.txt` (2 min read)
2. **Full analysis:** `DIAGNOSIS.md` (10 min read)
3. **Run diagnostics:** `./run_all_diagnostics.sh`

## File Guide

### Documentation

| File | Purpose | Read Time |
|------|---------|-----------|
| `QUICK_SUMMARY.txt` | TL;DR of problem, causes, and fixes | 2 min |
| `README.md` | Overview and script descriptions | 3 min |
| `DIAGNOSIS.md` | Complete technical analysis with evidence | 10 min |
| `INDEX.md` | This file - navigation guide | 1 min |

### Diagnostic Scripts

| Script | What It Tests | Runtime |
|--------|---------------|---------|
| `verify_maximally_mixed.py` | Loads checkpoint, confirms output is I/32 | ~30 sec |
| `analyze_training_curves.py` | Parses CSVs, shows training flatline | ~1 sec |
| `test_decoder_architecture.py` | Demonstrates decoder(enc, enc) flaw | ~1 sec |
| `test_output_mixing.py` | Shows per-row projection problem | ~1 sec |
| `gradient_flow_analysis.py` | Tests Cholesky layer gradient issues | ~1 sec |
| `compare_with_mlp.py` | Comprehensive architecture comparison | ~1 sec |

### Automation

| File | Purpose |
|------|---------|
| `run_all_diagnostics.sh` | Runs all 6 diagnostic scripts in sequence |

## Problem Summary

**Symptom:** Model outputs I/32 for all inputs, achieving constant ~0.031 fidelity

**Root Causes:**
1. `decoder(enc, enc)` - Degenerate autoencoder architecture
2. Per-row output projection - No global coordination for Cholesky
3. Discarded CLS token - Global aggregation unused

**Evidence:**
- Training: 0.823 → 0.823 (zero improvement)
- Fidelity: 0.031 ± 0.0002 (all noise types identical)
- 1/32 = 0.03125 ≈ 0.031 (maximally mixed state)

## Key Files in Main Repository

| Type | Path |
|------|------|
| Model | `/home/work/codage/transformer_qnr/models_3/transformer_cholesky.py` |
| MLP (working) | `/home/work/codage/transformer_qnr/models_3/mlp_cholesky.py` |
| Cholesky Layer | `/home/work/codage/transformer_qnr/models_3/cholesky_output.py` |
| Training Script | `/home/work/codage/transformer_qnr/train_cholesky.py` |
| Checkpoint | `/home/work/codage/transformer_qnr/checkpoints_3/transformer_cholesky/best.pt` |
| Training Log | `/home/work/codage/transformer_qnr/csvs_3/transformer_cholesky.csv` |
| MLP Log | `/home/work/codage/transformer_qnr/csvs_3/mlp_cholesky.csv` |
| Test Results | `/home/work/codage/transformer_qnr/csvs_3/noise_cells/transformer_cholesky_noise_cells.csv` |
| MLP Results | `/home/work/codage/transformer_qnr/csvs_3/noise_cells/mlp_cholesky_noise_cells.csv` |

## Reproduction Steps

### Option 1: Read Analysis
```bash
cd /home/work/codage/transformer_qnr/debug_transformer_collapse
cat QUICK_SUMMARY.txt              # 2 min overview
cat DIAGNOSIS.md | less            # Full technical analysis
```

### Option 2: Run Diagnostics
```bash
cd /home/work/codage/transformer_qnr/debug_transformer_collapse
./run_all_diagnostics.sh           # Runs all 6 scripts (~1 min total)
```

### Option 3: Run Individual Tests
```bash
cd /home/work/codage/transformer_qnr/debug_transformer_collapse

# Test 1: Verify maximally mixed output
python3 verify_maximally_mixed.py

# Test 2: Analyze training curves
python3 analyze_training_curves.py

# Test 3: Decoder architecture flaw
python3 test_decoder_architecture.py

# Test 4: Output mixing problem
python3 test_output_mixing.py

# Test 5: Gradient flow issues
python3 gradient_flow_analysis.py

# Test 6: Compare with MLP
python3 compare_with_mlp.py
```

## Key Findings

### Training Comparison
```
Model         Epoch 1    Best      Improvement
Transformer   0.8233     0.8237    0.0004  ← FLAT (no learning)
MLP           0.8331     0.7990    0.0341  ← Clear learning (85× more)
```

### Fidelity Comparison
```
Model         Range      Std Dev    Variation
Transformer   0.031      0.0002     None (constant across all noise)
MLP           0.03-0.07  0.02-0.06  High (varies with noise type/level)
```

### Maximally Mixed State
```
I/32 diagonal value:  0.03125
Observed fidelity:    0.031   ← Almost exactly 1/32!
```

## Recommended Fixes

See `DIAGNOSIS.md` section "Recommendations" for detailed fix options.

**Quick fix:** Use CLS token for global prediction
```python
# Replace lines 105-107 in transformer_cholesky.py
cls_output = dec[:, 0, :]  # (B, embed_dim) - USE CLS!
chol_params = nn.Linear(embed_dim, 1024)(cls_output)
```

## Architecture Comparison

### Transformer (BROKEN)
- 2048 → 2112 → 2048 → 1024 (no bottleneck)
- Per-row output projection (no global mixing)
- decoder(enc, enc) (redundant attention)
- Result: I/32 output, 0.031 fidelity

### MLP (WORKING)
- 2048 → 128 → 142 → 1024 (forced bottleneck)
- Global output projection (all params coordinated)
- Simple feed-forward (proven architecture)
- Result: Variable output, 0.03-0.07 fidelity

## Questions Answered

**Q: Why does it output exactly ~0.031?**
A: This is 1/32, the fidelity of the maximally mixed state I/32 with typical density matrices.

**Q: Why didn't training improve at all?**
A: Three architectural flaws (decoder, no global mixing, discarded CLS) create a degenerate landscape where I/32 is a stable fixed point.

**Q: Why does the MLP work?**
A: Forced bottleneck + global output projection allow all 1024 Cholesky params to coordinate.

**Q: Is the Cholesky layer broken?**
A: No, it works correctly. But when fed uniform inputs, it correctly produces I/32.

**Q: Is row tokenization the problem?**
A: Partly. Row tokenization is incompatible with Cholesky factorization, which requires global coordination. The per-row output projection makes this worse.

**Q: Can this be fixed?**
A: Yes, but requires architectural changes. See "Recommendations" in DIAGNOSIS.md.

## Lessons Learned

1. **Verify training early** - Flat loss from epoch 1 should trigger immediate investigation
2. **Avoid multiple innovations** - Row tokenization + Cholesky + no bottleneck = too many failure modes
3. **Ensure architectural compatibility** - Cholesky needs global coordination; row-wise processing prevents it
4. **Trust simple baselines** - MLP outperforms complex transformer here
5. **Use global information** - Discarding CLS token wastes the one component that could provide global mixing

## Total Analysis

- 10 files
- 1,276 lines of analysis and diagnostic code
- 6 reproducible diagnostic scripts
- Complete evidence trail from training logs to architectural flaws

---

**Start here:** Read `QUICK_SUMMARY.txt` then run `./run_all_diagnostics.sh`
