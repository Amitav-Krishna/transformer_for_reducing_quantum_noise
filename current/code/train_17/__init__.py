"""
train_17: 5-qubit training with Frobenius normalization.

Key change from train_v8: Normalize density matrices to unit Frobenius norm
before training to fix scale mismatch between noisy (~0.07 norm) and clean
(~1.0 norm) matrices.
"""
