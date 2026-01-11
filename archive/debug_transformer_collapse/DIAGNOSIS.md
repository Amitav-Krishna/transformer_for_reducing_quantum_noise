# Transformer Cholesky Model Collapse - Complete Diagnosis

## Executive Summary

The TransformerCholeskyAutoencoder has **completely collapsed** to outputting the maximally mixed state (I/32), achieving ~0.031 fidelity regardless of input with negligible variance (std 0.0002). Training never progressed beyond initialization, with validation loss remaining flat at ~0.823 across 24 epochs.

## Evidence of Collapse

### 1. Constant Output Regardless of Input
From `/home/work/codage/transformer_qnr/csvs_3/noise_cells/transformer_cholesky_noise_cells.csv`:

| Noise Type | Noise Level | Mean Fidelity | Std Fidelity |
|------------|-------------|---------------|--------------|
| amplitude_damping | 0.05-0.20 | 0.0313 | 0.0002 |
| bitflip | 0.05-0.20 | 0.0313 | 0.0002 |
| depolarizing | 0.05-0.20 | 0.0314 | 0.0003 |
| mixed | 0.05-0.20 | 0.0314 | 0.0003 |

**All noise types and levels produce identical output:** ~0.031 ≈ 1/32

### 2. Training Never Started
From `/home/work/codage/transformer_qnr/csvs_3/transformer_cholesky.csv`:

```
Epoch  1: Val loss = 0.823346
Epoch  5: Val loss = 0.823406
Epoch 10: Val loss = 0.823916
Epoch 15: Val loss = 0.823423
Epoch 20: Val loss = 0.823692
Epoch 24: Val loss = 0.823680 (early stopped)

Total improvement: 0.0003 (effectively ZERO)
```

Compare to MLP Cholesky:
```
Epoch  1: Val loss = 0.833083
Epoch 10: Val loss = 0.817082
Epoch 30: Val loss = 0.803864
Epoch 96: Val loss = 0.799012 (best)

Total improvement: 0.034 (113× more than transformer)
```

### 3. MLP Shows Variable Learning
From `/home/work/codage/transformer_qnr/csvs_3/noise_cells/mlp_cholesky_noise_cells.csv`:

| Noise Type | Noise Level | Mean Fidelity | Std Fidelity |
|------------|-------------|---------------|--------------|
| amplitude_damping | 0.05 | 0.0562 | 0.0409 |
| mixed | 0.10 | 0.0671 | 0.0562 |
| bitflip | 0.15 | 0.0297 | 0.0237 |

**Fidelity varies meaningfully:** 0.03-0.07 range with high std dev (0.02-0.06)

## Root Cause Analysis

### Critical Flaw #1: Degenerate Decoder Architecture

**Location:** `/home/work/codage/transformer_qnr/models_3/transformer_cholesky.py`, line 102

```python
# Encode
enc = self.encoder(x)

# Decode (no bottleneck - direct cross-attention to encoder output)
dec = self.decoder(enc, enc)  # ← FLAW: tgt == memory
```

**What happens:** `TransformerDecoder(tgt, memory)` performs:
1. Self-attention: `tgt` attends to itself (enc attends to enc)
2. Cross-attention: result attends to `memory` (which is ALSO enc)

**Why this is broken:**
- This is NOT how autoencoders work
- No bottleneck forces information compression
- Redundant attention: attending to the same tensor twice through different mechanisms
- Model can learn identity mapping or uniform output with minimal loss

**Correct approaches:**
- Use learned query tokens: `decoder(learned_queries, enc)`
- Use encoder-only: `encoder(x) → bottleneck → encoder(bottleneck)`
- Use separate encoder for decoding: `decoder_encoder(enc)`

### Critical Flaw #2: No Global Mixing in Output Layer

**Location:** `/home/work/codage/transformer_qnr/models_3/transformer_cholesky.py`, lines 104-107

```python
# Extract row tokens (skip CLS at position 0), project to Cholesky params
row_output = dec[:, 1:, :]  # (B, 32, embed_dim) - skip CLS
chol_params = self.output_proj(row_output)  # (B, 32, 32)
chol_params = chol_params.reshape(B, -1)  # (B, 1024)
```

**What `self.output_proj = nn.Linear(embed_dim, 32)` does:**
```
Row 0 output = Linear(row_0_embedding) → 32 values
Row 1 output = Linear(row_1_embedding) → 32 values
...
Row 31 output = Linear(row_31_embedding) → 32 values
```

**Why this breaks Cholesky factorization:**

The Cholesky factorization computes ρ = LL† where:
- ρ[i,j] = Σ_k L[i,k] × L[j,k]*
- Matrix element (i,j) depends on the dot product of row i and row j of L

If L[i,:] and L[j,:] are computed **independently** from different embeddings, they cannot coordinate to produce the required correlations in ρ.

**Result:** Model learns degenerate solution where all rows are uniform → L ≈ I/√32 → ρ = LL† ≈ I/32

### Critical Flaw #3: CLS Token Discarded

**Location:** Line 105: `row_output = dec[:, 1:, :]  # skip CLS`

The architecture:
1. Creates a learnable CLS token (line 46)
2. Allows it to aggregate global information through attention
3. **Immediately throws it away** before reconstruction

**Impact:**
- CLS token could provide global mixing needed for Cholesky coordination
- Instead, only per-row embeddings are used
- The one component that could save the architecture is discarded

### Secondary Issue: No Effective Bottleneck

**Architecture flow:**
```
Input: (B, 2, 32, 32) = 2048 values
↓
Row tokenization: (B, 32, 64) = 2048 values
↓
Add CLS: (B, 33, 64) = 2112 values  ← EXPANSION, not compression!
↓
Encoder: (B, 33, 64) → (B, 33, 64)
↓
Decoder: (B, 33, 64) → (B, 33, 64)
↓
Drop CLS: (B, 32, 64) = 2048 values
↓
Output: (B, 1024)
```

**Problem:** Information never goes through a true bottleneck:
- 2048 → 2112 → 2048 → 1024
- Model can route information through without compression
- Compare to MLP: 2048 → 128 → 142 → 1024 (forced 16× compression)

### Gradient Flow Issues

**Location:** `/home/work/codage/transformer_qnr/models_3/cholesky_output.py`, lines 54, 76

```python
# Diagonal: real and positive via softplus with minimum floor
diag = F.softplus(diag_raw) + 1e-4

# Normalize trace to 1
rho = rho / (trace.real + 1e-8)
```

**When model outputs uniform values:**
1. If params ≈ 0, then softplus(0) + 1e-4 ≈ 0.693 + 1e-4 ≈ 0.693
2. If params ≈ -5, then softplus(-5) + 1e-4 ≈ 0.007 + 1e-4 ≈ 0.007
3. With negative learned weights, diagonal → 1e-4
4. Trace ≈ 32 × (1e-4)² ≈ 3.2e-7
5. After normalization: ρ ≈ I/32

**Issue:** This is a **stable fixed point**:
- Gradients through division by tiny numbers
- Model has no incentive to move away from uniform output
- Loss landscape has flat basin around I/32

## Why MLP Succeeds Where Transformer Fails

### MLP Architecture (WORKING)
```python
nn.Linear(2048, 128),   # Global compression (16× reduction)
nn.ReLU(),
nn.Dropout(0.1),
nn.Linear(128, 142),    # Bottleneck
nn.ReLU(),
nn.Dropout(0.1),
nn.Linear(142, 1024)    # All 1024 params from same representation
```

**Advantages:**
1. **Global mixing:** `Linear(142, 1024)` means every output depends on entire bottleneck
2. **Forced compression:** Must compress 2048 → 128 (cannot route around it)
3. **Simple gradient path:** Only 3 linear layers
4. **Proven architecture:** Standard MLP autoencoder design

**Key insight:** When computing `output = W @ bottleneck`:
- Output[i] depends on bottleneck via weights W[i,:]
- Output[j] depends on same bottleneck via weights W[j,:]
- Both outputs can coordinate because they share the same bottleneck
- This coordination is ESSENTIAL for Cholesky factorization

### Transformer Architecture (BROKEN)

**Disadvantages:**
1. **No global mixing:** Each row's output computed independently
2. **No bottleneck:** 2048 → 2112 → 2048 (expansion then reduction)
3. **Complex gradient path:** 10 transformer layers + multiple projections
4. **Novel architecture:** Multiple simultaneous innovations = more failure modes

## Comparison: Why Row Tokenization Fails Here

**The fundamental incompatibility:**

Row-based tokenization is elegant for:
- Sequence-to-sequence tasks
- Tasks where rows can be processed independently
- Vision transformers where patches are somewhat independent

But Cholesky factorization requires:
- Global coordination of ALL 1024 parameters
- Element ρ[i,j] depends on L[i,:] • L[j,:]
- Rows i and j of L must be computed jointly to create correlations

**The MLP computes all params jointly. The Transformer computes them separately.**

## Verification Scripts

All scripts in `/home/work/codage/transformer_qnr/debug_transformer_collapse/`:

1. `verify_maximally_mixed.py` - Loads checkpoint, verifies output is I/32
2. `analyze_training_curves.py` - Shows training flatline vs MLP improvement
3. `test_decoder_architecture.py` - Demonstrates decoder(enc, enc) flaw
4. `test_output_mixing.py` - Shows lack of global coordination
5. `gradient_flow_analysis.py` - Analyzes Cholesky layer gradient issues
6. `compare_with_mlp.py` - Comprehensive architecture comparison

## Recommendations

### For Fixing This Model

**Option 1: Use CLS token for global prediction (minimal change)**
```python
# Instead of:
row_output = dec[:, 1:, :]  # Skip CLS
chol_params = self.output_proj(row_output)  # Per-row projection

# Do:
cls_output = dec[:, 0, :]  # (B, embed_dim) - USE CLS!
global_proj = nn.Linear(embed_dim, 1024)
chol_params = global_proj(cls_output)  # (B, 1024) - global prediction
```

**Option 2: Fix decoder architecture**
```python
# Instead of:
dec = self.decoder(enc, enc)

# Use learned queries:
self.dec_queries = nn.Parameter(torch.randn(1, num_tokens, embed_dim))
queries = self.dec_queries.expand(B, -1, -1)
dec = self.decoder(queries, enc)
```

**Option 3: Use encoder-only with proper bottleneck**
```python
enc = self.encoder(x)  # (B, 33, 64)
# Pool to bottleneck
bottleneck = enc.mean(dim=1)  # (B, 64) or use attention pooling
# Expand back
expanded = self.expand(bottleneck).view(B, 33, 64)
# Second encoder for decoding
dec = self.decoder_encoder(expanded)
```

### General Lessons

1. **Don't innovate on multiple axes simultaneously**
   - Row tokenization: innovative
   - Cholesky output: innovative
   - No bottleneck: questionable
   - Result: Too many failure modes

2. **Ensure architectural compatibility**
   - Cholesky requires global coordination
   - Row-wise processing prevents global coordination
   - These are fundamentally incompatible

3. **Trust simple baselines**
   - MLP with ~same params performs better
   - Sometimes simplicity is better than cleverness

4. **Always verify training is working**
   - Flat loss from epoch 1 = model never learned
   - Should have stopped and debugged immediately

## Files Reference

| File | Path |
|------|------|
| Model | `/home/work/codage/transformer_qnr/models_3/transformer_cholesky.py` |
| Training | `/home/work/codage/transformer_qnr/train_cholesky.py` |
| Cholesky Layer | `/home/work/codage/transformer_qnr/models_3/cholesky_output.py` |
| Checkpoint | `/home/work/codage/transformer_qnr/checkpoints_3/transformer_cholesky/best.pt` |
| Training Log | `/home/work/codage/transformer_qnr/csvs_3/transformer_cholesky.csv` |
| Noise Cell Results | `/home/work/codage/transformer_qnr/csvs_3/noise_cells/transformer_cholesky_noise_cells.csv` |
| MLP (comparison) | `/home/work/codage/transformer_qnr/models_3/mlp_cholesky.py` |

## Conclusion

The Transformer Cholesky model failed due to **three critical architectural flaws working in concert**:

1. **Degenerate decoder**: `decoder(enc, enc)` creates redundant attention with no bottleneck
2. **No global mixing**: Per-row output projection prevents Cholesky coordination
3. **Discarded CLS token**: The one global component is thrown away

The result is a model that learns to output the maximally mixed state (I/32) as the path of least resistance, achieving constant ~0.031 fidelity regardless of input.

The MLP succeeds because it forces global mixing through a bottleneck, allowing all 1024 Cholesky parameters to coordinate.

**The fundamental lesson:** Row-based tokenization, while elegant, is **incompatible with Cholesky factorization**, which requires global coordination of all parameters.
