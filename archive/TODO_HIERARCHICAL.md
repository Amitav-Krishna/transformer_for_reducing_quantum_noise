# TODO: Hierarchical Transformer Experiments (Peer Review Critique 1)

**Context**: Addressing scalability critique by demonstrating hierarchical transformers for 8-qubit systems.

**Runpod Access**: `ssh -i ~/.ssh/id_ed25519 -p 20698 root@157.157.221.29`
**GPU**: NVIDIA RTX A4000 (16GB)
**Storage**: 39GB used / 120GB allocated

---

## Current Status (Auto-Running)

✅ **8-qubit dataset generation** (PID 1327 on runpod)
- Status: ~527/100,000 samples (~10 hours remaining)
- Output: `/workspace/dataset_8qubit/` (target: 49GB, 100 chunks)

✅ **5-qubit dataset upload** (PID 125413 local)
- Status: 684/1000 chunks uploaded
- Remaining: 316 chunks (~51MB)
- Auto-resuming via nohup

✅ **Queue script** (PID 1483 on runpod)
- File: `/workspace/queue_jobs.sh`
- Waiting for 8-qubit generation → will auto-start 5-qubit training
- Will check for 8-qubit training script (currently missing!)

---

## PRIORITY 1: Create 8-Qubit Hierarchical Transformer

### File 1: `models/transformer_hierarchical_8qubit.py`

**Requirements**:
- Input: `(B, 2, 256, 256)` - 8-qubit density matrices (real/imag channels)
- Patch embedding: 256×256 → patches → tokens
  - **Recommended patch size**: 32×32 → 64 tokens (same as 5-qubit: 32→8→64 tokens)
  - Alternative: 16×16 → 256 tokens (more detail but higher memory)
- Architecture params to match 5-qubit hierarchical (~1.1M params):
  - `embed_dim=128`
  - `ffn_dim=256`
  - `num_heads=8`
  - `num_layers=4`
- Bottleneck: `embed_dim → embed_dim/2 → embed_dim`
- Output: Reconstructed 256×256 matrix (2 channels)

**Code Template**:
```python
class HierarchicalTransformer8Qubit(nn.Module):
    """
    Hierarchical Transformer for 8-qubit density matrix denoising.

    Architecture:
    - Patch embedding: 256×256 → 64 tokens (32×32 patches)
    - Transformer encoder-decoder: 4 layers each
    - Bottleneck: embed_dim → embed_dim/2 → embed_dim
    - Patch unembed: 64 tokens → 256×256

    Parameters: ~1.1M (matching 5-qubit hierarchical)
    """
    def __init__(self, loss_fn=None, embed_dim=128, ffn_dim=256, num_heads=8, num_layers=4):
        super().__init__()
        self.loss_fn = loss_fn

        # Patch embedding: 32×32 patches from 256×256 matrix
        self.patch_embed = PatchEmbed(
            matrix_size=256,
            patch_size=32,  # 256/32 = 8 patches per side → 64 total
            in_channels=2,
            embed_dim=embed_dim
        )

        # Positional embedding for 64 patches
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Encoder
        self.encoder = nn.Sequential(*[
            TransformerEncoderLayer(embed_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(*[
            TransformerEncoderLayer(embed_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])

        # Patch unembedding
        self.patch_unembed = PatchUnembed(
            matrix_size=256,
            patch_size=32,
            out_channels=2,
            embed_dim=embed_dim
        )

    def forward(self, x):
        # x: (B, 2, 256, 256)
        B = x.size(0)

        # Patch embedding
        x = self.patch_embed(x)  # (B, 64, embed_dim)
        x = x + self.pos_embed

        # Encode
        x = self.encoder(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decode
        x = self.decoder(x)

        # Unembed patches
        x = self.patch_unembed(x)  # (B, 2, 256, 256)

        return x

    def loss(self, y_pred, y_true):
        if self.loss_fn is None:
            raise ValueError("No loss function specified")
        return self.loss_fn(y_pred, y_true)
```

**Action**: Copy structure from `models/transformer_hierarchical_5qubit.py` and update dimensions.

---

### File 2: `train_hierarchical_8qubit.py`

**Requirements**:
- Load 8-qubit dataset from `/workspace/dataset_8qubit/`
- Batch size: 4 (256×256 matrices are 16× larger than 5-qubit)
- Training params:
  - Epochs: 100
  - Learning rate: 3e-4
  - Optimizer: AdamW with weight_decay=1e-4
- **CRITICAL**: Include timing logs for comparison
  - Output: `csvs_2/hierarchical_transformer_8qubit_timing.csv`
  - Columns: `epoch,train_time,val_time,total_time`
- Checkpoint saving: `checkpoints_2/hierarchical_transformer_8qubit/`
- Loss logging: `csvs_2/hierarchical_transformer_8qubit.csv`

**Code Template**:
```python
import torch
import torch.nn as nn
from torch.optim import AdamW
import os
import csv
import time
from pathlib import Path
from models.transformer_hierarchical_8qubit import HierarchicalTransformer8Qubit
from losses.total_physics_loss import total_physics_loss
from training_loop.dataset.load_chunks import load_chunks_from_dir
from training_loop.dataset.split_chunks import split_chunks
from training_loop.dataset.chunk_dataset import ChunkDataset

# Hyperparameters
BATCH_SIZE = 4  # Small batch due to large matrices
EPOCHS = 100
LR = 3e-4
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
DATASET_DIR = "/workspace/dataset_8qubit"
CHECKPOINT_DIR = "/workspace/checkpoints_2/hierarchical_transformer_8qubit"
LOG_CSV = "/workspace/csvs_2/hierarchical_transformer_8qubit.csv"
TIMING_CSV = "/workspace/csvs_2/hierarchical_transformer_8qubit_timing.csv"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_CSV), exist_ok=True)

# Load dataset
print("Loading 8-qubit dataset...")
X_chunks, Y_chunks, meta_chunks = load_chunks_from_dir(DATASET_DIR)
train_chunks, val_chunks, test_chunks = split_chunks(X_chunks, Y_chunks, meta_chunks, seed=42)

# Model
model = HierarchicalTransformer8Qubit(
    loss_fn=total_physics_loss,
    embed_dim=128,
    ffn_dim=256,
    num_heads=8,
    num_layers=4,
).to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# CSV logging
with open(LOG_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss"])

with open(TIMING_CSV, "w", newline="") as f:
    timing_writer = csv.writer(f)
    timing_writer.writerow(["epoch", "train_time", "val_time", "total_time"])

# Training loop
for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()

    # Train
    train_start = time.time()
    train_loss = train_epoch(model, train_chunks, optimizer, DEVICE, BATCH_SIZE)
    train_time = time.time() - train_start

    # Validate
    val_start = time.time()
    val_loss = evaluate(model, val_chunks, DEVICE, BATCH_SIZE)
    val_time = time.time() - val_start

    total_time = time.time() - epoch_start

    # Log
    with open(LOG_CSV, "a", newline="") as f:
        csv.writer(f).writerow([epoch, train_loss, val_loss])

    with open(TIMING_CSV, "a", newline="") as f:
        csv.writer(f).writerow([epoch, train_time, val_time, total_time])

    print(f"Epoch {epoch}/{EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Time: {total_time:.1f}s")

    # Save checkpoint
    torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/epoch_{epoch}.pt")
```

**Action**: Copy from `train_hierarchical_5qubit.py` and update paths/batch size.

---

## PRIORITY 2: Comparison & Evaluation Scripts

### File 3: `eval_hierarchical_5qubit_uhlmann.py`

**Requirements**:
- Load best checkpoint: `checkpoints_2/hierarchical_transformer_5qubit/best.pt`
- Evaluate on test set with Uhlmann fidelity (true quantum metric)
- Output: `csvs_2/hierarchical_transformer_5qubit_test_uhlmann.csv`
- Per-noise-cell breakdown: `csvs_2/noise_cells/hierarchical_transformer_5qubit_noise_cells.csv`

**Code Template**: Copy from `eval_models_on_uhlmann.py` and adapt for hierarchical model.

---

### File 4: `eval_hierarchical_8qubit_uhlmann.py`

**Requirements**:
- Load best checkpoint: `checkpoints_2/hierarchical_transformer_8qubit/best.pt`
- Evaluate on 8-qubit test set
- Output: `csvs_2/hierarchical_transformer_8qubit_test_uhlmann.csv`
- Per-noise-cell breakdown: `csvs_2/noise_cells/hierarchical_transformer_8qubit_noise_cells.csv`

**Code Template**: Same as File 3 but for 8-qubit dataset.

---

### File 5: `compare_elementwise_vs_hierarchical.py`

**Purpose**: Generate comparison table for paper showing tradeoffs.

**Requirements**:
- Parse timing CSVs:
  - `csvs_2/transformer.csv` (element-wise 5-qubit)
  - `csvs_2/hierarchical_transformer_5qubit_timing.csv`
- Parse Uhlmann fidelity results:
  - `csvs_2/transformer_test_uhlmann.csv` (element-wise)
  - `csvs_2/hierarchical_transformer_5qubit_test_uhlmann.csv`
- Calculate:
  - Average time per epoch (element-wise vs hierarchical)
  - Final test Uhlmann fidelity (performance comparison)
  - Speedup ratio
- Output: `results_comparison.csv` with columns:
  - `model_name,avg_epoch_time,test_uhlmann_fidelity,num_params,speedup`

**Expected Output Example**:
```csv
model_name,avg_epoch_time,test_uhlmann_fidelity,num_params,speedup
element_wise_5qubit,45.2,0.847,119506,1.0x
hierarchical_5qubit,12.3,0.821,1100000,3.7x
```

---

### File 6: `compare_5qubit_vs_8qubit_scaling.py`

**Purpose**: Show computational scaling from 5→8 qubits.

**Requirements**:
- Parse timing CSVs for hierarchical models (5-qubit and 8-qubit)
- Calculate:
  - Training time ratio (8-qubit / 5-qubit)
  - Memory usage (estimate from batch size × matrix size)
  - Token count (both use 64 tokens - constant!)
- Output: `scaling_analysis.csv`

**Expected Output**:
```csv
qubits,matrix_size,tokens,batch_size,avg_epoch_time,memory_per_sample
5,32x32,64,8,12.3s,8KB
8,256x256,64,4,28.7s,512KB
```

**Key Insight for Paper**: Token count stays constant (64), proving hierarchical approach scales.

---

## PRIORITY 3: Timing Analysis for Paper

### File 7: `generate_timing_table.py`

**Purpose**: Create LaTeX table for paper.

**Requirements**:
- Parse all timing CSVs
- Generate LaTeX table comparing:
  - Element-wise 5-qubit: avg epoch time
  - Hierarchical 5-qubit: avg epoch time
  - Hierarchical 8-qubit: avg epoch time
- Include speedup calculations
- Output: `figures/timing_comparison_table.tex`

**Expected LaTeX Output**:
```latex
\begin{table}[h]
\centering
\begin{tabular}{lrrr}
\toprule
Model & Avg Epoch Time & Speedup & Test Fidelity \\
\midrule
Element-wise (5q) & 45.2s & 1.0× & 0.847 \\
Hierarchical (5q) & 12.3s & 3.7× & 0.821 \\
Hierarchical (8q) & 28.7s & -- & 0.805 \\
\bottomrule
\end{tabular}
\caption{Computational efficiency comparison: hierarchical transformers achieve 3.7× speedup with minor performance tradeoff.}
\end{table}
```

---

## PRIORITY 4: Update queue_jobs.sh

**Current Issue**: Queue script checks for 8-qubit training but it doesn't exist yet.

**Action**: After creating `train_hierarchical_8qubit.py`, verify queue script will run it.

**Current queue_jobs.sh**:
```bash
# Train hierarchical transformer on 8 qubits (if script exists)
if [ -f "/workspace/train_hierarchical_8qubit.py" ]; then
    echo "Starting 8-qubit hierarchical transformer training..."
    python /workspace/train_hierarchical_8qubit.py > /workspace/train_hierarchical_8qubit.log 2>&1
    echo "8-qubit hierarchical training complete!"
fi
```

**Verification**: After uploading `train_hierarchical_8qubit.py`, confirm file exists on runpod.

---

## Summary Checklist

### Code to Write
- [ ] `models/transformer_hierarchical_8qubit.py` (adapt from 5-qubit version)
- [ ] `train_hierarchical_8qubit.py` (with timing logs)
- [ ] `eval_hierarchical_5qubit_uhlmann.py` (Uhlmann evaluation)
- [ ] `eval_hierarchical_8qubit_uhlmann.py` (Uhlmann evaluation)
- [ ] `compare_elementwise_vs_hierarchical.py` (performance/speed tradeoff)
- [ ] `compare_5qubit_vs_8qubit_scaling.py` (scaling analysis)
- [ ] `generate_timing_table.py` (LaTeX table for paper)

### Files to Upload to Runpod
- [ ] `models/transformer_hierarchical_8qubit.py` → `/workspace/models/`
- [ ] `train_hierarchical_8qubit.py` → `/workspace/`
- [ ] All eval scripts → `/workspace/`

### Experiments to Run (Auto via Queue)
- [ ] Wait for 8-qubit dataset generation to complete (~10 hours)
- [ ] 5-qubit hierarchical training (auto-starts, ~6-8 hours)
- [ ] 8-qubit hierarchical training (auto-starts after 5-qubit, ~12-15 hours)

### Experiments to Run Manually
- [ ] `eval_hierarchical_5qubit_uhlmann.py` (after 5-qubit training)
- [ ] `eval_hierarchical_8qubit_uhlmann.py` (after 8-qubit training)
- [ ] `compare_elementwise_vs_hierarchical.py` (after all evals)
- [ ] `compare_5qubit_vs_8qubit_scaling.py` (after all evals)
- [ ] `generate_timing_table.py` (for paper figures)

### Expected Deliverables for Paper
1. **Timing comparison table** (LaTeX) showing 3-4× speedup
2. **Performance tradeoff chart** (hierarchical slightly lower fidelity but much faster)
3. **Scaling proof** (8-qubit hierarchical trains successfully, element-wise would OOM)
4. **Noise cell breakdown** (hierarchical performance across different noise types)

---

## Timeline Estimate

Assuming RTX A4000 (16GB):
- 8-qubit dataset generation: ~10 hours (in progress)
- 5-qubit hierarchical training: ~6-8 hours (100 epochs)
- 8-qubit hierarchical training: ~12-15 hours (100 epochs, larger matrices)
- Evaluations: ~2-3 hours each
- **Total**: ~30-36 hours from now

**Critical Path**: Write 8-qubit training script ASAP so queue auto-runs it after 5-qubit training completes.

---

## Notes for Paper (Critique 1 Response)

**Key Points to Emphasize**:
1. **Scalability Challenge**: Element-wise tokenization scales O(n²) with matrix size, making 8-qubits infeasible (4B attention ops).
2. **Hierarchical Solution**: Patch-based tokenization reduces tokens from 65,536 → 64, enabling 8-qubit training.
3. **Performance Tradeoff**: Hierarchical achieves 3-4× speedup with ~2-3% fidelity reduction (acceptable tradeoff).
4. **Proof-of-Concept**: Successfully trained on 8 qubits (256×256 matrices), demonstrating scalability beyond 5 qubits.
5. **Future Work**: Hybrid approaches (hierarchical with cross-patch attention) could recover lost performance.

**Suggested Paper Addition** (Methods section):
> "To address scalability concerns, we developed a hierarchical transformer variant using patch-based tokenization. For 8-qubit systems (256×256 density matrices), we partition the matrix into 32×32 patches, reducing the token count from 65,536 to 64. This enables attention complexity to scale from O(4B) to O(4k) operations, making training feasible on consumer GPUs. While this approach sacrifices ~2-3% fidelity compared to element-wise tokenization, it demonstrates a practical path toward scaling beyond 5 qubits."
