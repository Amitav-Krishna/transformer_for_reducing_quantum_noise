"""
train_15: Hierarchical MLP for 8-qubit density matrices.

Control experiment comparing hierarchical MLP vs hierarchical Transformer
on 8-qubit data. Both use identical patch embedding (32x32 patches -> 64 tokens),
differing only in the mixing mechanism:
- Transformer: attention-based mixing between patches
- MLP: feedforward mixing between patches

This isolates the attention vs feedforward comparison at scale.
"""
