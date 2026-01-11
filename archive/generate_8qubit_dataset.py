"""
Generate 8-qubit dataset for scalability proof-of-concept.

Circuit structure (matching 5-qubit dataset pattern):
- Depth = 6-9 layers (random per circuit)
- Each layer: single-qubit gates on ALL 8 qubits + ~4 two-qubit gates
- Total: ~48-72 single-qubit gates + ~24-36 two-qubit gates per circuit

This creates significant entanglement and allows noise to accumulate,
similar to the 5-qubit dataset.

Output: 256×256 density matrices (65,536 elements)
Target: 100,000 samples (~34 GB)

Usage:
    python generate_8qubit_dataset.py
"""

import os
import cirq
import numpy as np
import random
import torch
from tqdm import tqdm


def generate_circuit_8qubit(num_qubits=8, depth=None, seed=None):
    """
    Generate an 8-qubit circuit with specified depth.

    Each depth layer contains:
    - Single-qubit gates on ALL qubits
    - ~4 two-qubit gates on random adjacent pairs

    This matches the 5-qubit dataset generation pattern.

    Args:
        num_qubits: Number of qubits (default 8)
        depth: Circuit depth (default: random 6-9)
        seed: Random seed for reproducibility

    Returns:
        tuple: (circuit, qubits) - both the circuit and all qubits for simulation
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    if depth is None:
        depth = random.randint(6, 9)

    qubits = cirq.LineQubit.range(num_qubits)
    circuit = cirq.Circuit()

    # Single qubit gates to sample from
    single_qubit_gates = [
        cirq.X,
        cirq.Y,
        cirq.Z,
        cirq.H,
        cirq.T,
        cirq.S,
        lambda q: cirq.rx(np.random.uniform(0, 2 * np.pi))(q),
        lambda q: cirq.ry(np.random.uniform(0, 2 * np.pi))(q),
        lambda q: cirq.rz(np.random.uniform(0, 2 * np.pi))(q),
    ]

    two_qubit_gates = [
        cirq.CZ,
        cirq.CNOT,
        cirq.SWAP,
    ]

    # Build circuit layer by layer (matching 5-qubit pattern)
    qubit_pairs = list(zip(qubits[:-1], qubits[1:]))  # Adjacent pairs

    for d in range(depth):
        # Single-qubit gates on ALL qubits
        for q in qubits:
            gate = random.choice(single_qubit_gates)
            circuit.append(gate(q))

        # Two-qubit gates on random adjacent pairs (~half the pairs)
        n_two_qubit = min(len(qubit_pairs), num_qubits // 2)
        for q1, q2 in random.sample(qubit_pairs, k=n_two_qubit):
            gate = random.choice(two_qubit_gates)
            circuit.append(gate(q1, q2))

    return circuit, qubits


def add_noise(circuit, noise_type, noise_level):
    """Add specified noise model to circuit."""
    qubits = sorted(circuit.all_qubits())
    noisy_circuit = cirq.Circuit()

    if noise_type == "depolarizing":
        noise_model = cirq.depolarize(p=noise_level)
    elif noise_type == "amplitude_damping":
        noise_model = cirq.amplitude_damp(gamma=noise_level)
    elif noise_type == "phase_damping":
        noise_model = cirq.phase_damp(gamma=noise_level)
    elif noise_type == "bitflip":
        noise_model = cirq.bit_flip(p=noise_level)
    elif noise_type == "mixed":
        # Apply all noise types with 1/4 strength each
        for moment in circuit:
            noisy_circuit.append(moment)
            noisy_circuit.append(cirq.depolarize(p=noise_level/4).on_each(*qubits))
            noisy_circuit.append(cirq.amplitude_damp(gamma=noise_level/4).on_each(*qubits))
            noisy_circuit.append(cirq.phase_damp(gamma=noise_level/4).on_each(*qubits))
            noisy_circuit.append(cirq.bit_flip(p=noise_level/4).on_each(*qubits))
        return noisy_circuit
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    for moment in circuit:
        noisy_circuit.append(moment)
        noisy_circuit.append(noise_model.on_each(*qubits))

    return noisy_circuit


def generate_8qubit_dataset(
    output_dir="dataset_8qubit",
    num_samples_per_cell=5000,  # 5000 * 20 cells = 100,000 total
    noise_types=None,
    noise_levels=None,
    chunk_size=1000,
    seed=42,
):
    """
    Generate 8-qubit dataset and stream to disk.

    With 5 noise types × 4 noise levels × 5000 samples = 100,000 total.
    """
    os.makedirs(output_dir, exist_ok=True)

    if noise_types is None:
        noise_types = ["depolarizing", "amplitude_damping", "phase_damping", "bitflip", "mixed"]
    if noise_levels is None:
        noise_levels = [0.05, 0.10, 0.15, 0.20]

    np.random.seed(seed)
    random.seed(seed)

    simulator = cirq.DensityMatrixSimulator()
    total_samples = num_samples_per_cell * len(noise_types) * len(noise_levels)

    print(f"Generating {total_samples} 8-qubit samples...")
    print(f"Matrix size: 256×256 (65,536 elements)")
    print(f"Output directory: {output_dir}")

    chunk_counter = 0
    X_chunk, Y_chunk, meta_chunk = [], [], []

    with tqdm(total=total_samples) as pbar:
        for noise_type in noise_types:
            for noise_level in noise_levels:
                for i in range(num_samples_per_cell):
                    circuit, qubits = generate_circuit_8qubit(num_qubits=8)
                    # qubits are already LineQubit.range(8) in order

                    # Clean simulation
                    rho_clean = simulator.simulate(circuit, qubit_order=qubits).final_density_matrix

                    # Noisy simulation
                    noisy_circuit = add_noise(circuit, noise_type, noise_level)
                    rho_noisy = simulator.simulate(noisy_circuit, qubit_order=qubits).final_density_matrix

                    # Stack real/imag (shape: 256, 256, 2)
                    x_sample = np.stack([rho_noisy.real, rho_noisy.imag], axis=-1).astype(np.float32)
                    y_sample = np.stack([rho_clean.real, rho_clean.imag], axis=-1).astype(np.float32)

                    # Verify shape
                    assert x_sample.shape == (256, 256, 2), f"Bad X shape: {x_sample.shape}"
                    assert y_sample.shape == (256, 256, 2), f"Bad Y shape: {y_sample.shape}"

                    X_chunk.append(x_sample)
                    Y_chunk.append(y_sample)
                    meta_chunk.append({
                        "noise_type": noise_type,
                        "noise_level": noise_level,
                    })

                    pbar.update(1)

                    # Save chunk
                    if len(X_chunk) >= chunk_size:
                        chunk_counter += 1
                        fname = f"{output_dir}/chunk_{chunk_counter:04d}.pt"
                        torch.save({
                            "X": torch.from_numpy(np.stack(X_chunk, axis=0)),
                            "Y": torch.from_numpy(np.stack(Y_chunk, axis=0)),
                            "meta": meta_chunk,
                        }, fname)
                        print(f"Saved {fname} ({len(X_chunk)} samples)")
                        X_chunk, Y_chunk, meta_chunk = [], [], []

    # Save remainder
    if len(X_chunk) > 0:
        chunk_counter += 1
        fname = f"{output_dir}/chunk_{chunk_counter:04d}.pt"
        torch.save({
            "X": torch.from_numpy(np.stack(X_chunk, axis=0)),
            "Y": torch.from_numpy(np.stack(Y_chunk, axis=0)),
            "meta": meta_chunk,
        }, fname)
        print(f"Saved final {fname} ({len(X_chunk)} samples)")

    print(f"\nDataset generation complete!")
    print(f"Total samples: {total_samples}")
    print(f"Total chunks: {chunk_counter}")
    print(f"Output: {output_dir}/")


if __name__ == "__main__":
    generate_8qubit_dataset(
        output_dir="dataset_8qubit",
        num_samples_per_cell=5000,  # 5000 * 5 noise types * 4 levels = 100,000
        chunk_size=1000,
        seed=42,
    )
