#!/usr/bin/env python3
"""
Generate 5-qubit float64 dataset for the train_16 benchmark suite.

Key changes from original generate_dataset.py:
- Uses complex128 simulator: cirq.DensityMatrixSimulator(dtype=np.complex128)
- Saves as float64: torch.float64
- Verifies numerical precision end-to-end

Output: 32x32 density matrices in float64
Target: 100,000 samples (5 noise types x 4 levels x 5000 per cell)
"""

import os
import sys
import cirq
import numpy as np
import random
import torch
from tqdm import tqdm


# =============================================================================
# Circuit Generation (unchanged from original)
# =============================================================================


def generate_random_circuit(num_qubits, depth, seed=None):
    """Generate a random quantum circuit with specified number of qubits and depth."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    qubits = cirq.LineQubit.range(num_qubits)
    circuit = cirq.Circuit()

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
        lambda q1, q2: cirq.SWAP(q1, q2),
    ]

    for d in range(depth):
        # Single-qubit gates on all qubits
        for q in qubits:
            gate = random.choice(single_qubit_gates)
            circuit.append(gate(q))

        # Two-qubit gates on adjacent pairs
        qubit_pairs = list(zip(qubits[:-1], qubits[1:]))
        for q1, q2 in random.sample(
            qubit_pairs, k=min(len(qubit_pairs), num_qubits // 2)
        ):
            gate = random.choice(two_qubit_gates)
            circuit.append(gate(q1, q2))

    return circuit


def add_noise(circuit, noise_type, noise_level):
    """Add specified noise model to circuit."""
    qubits = sorted(circuit.all_qubits())
    noisy_circuit = cirq.Circuit()

    if noise_type == "depolarizing":
        noise_model = cirq.depolarize(p=noise_level)
        for moment in circuit:
            noisy_circuit.append(moment)
            noisy_circuit.append(noise_model.on_each(*qubits))
    elif noise_type == "amplitude_damping":
        noise_model = cirq.amplitude_damp(gamma=noise_level)
        for moment in circuit:
            noisy_circuit.append(moment)
            noisy_circuit.append(noise_model.on_each(*qubits))
    elif noise_type == "phase_damping":
        noise_model = cirq.phase_damp(gamma=noise_level)
        for moment in circuit:
            noisy_circuit.append(moment)
            noisy_circuit.append(noise_model.on_each(*qubits))
    elif noise_type == "bitflip":
        noise_model = cirq.bit_flip(p=noise_level)
        for moment in circuit:
            noisy_circuit.append(moment)
            noisy_circuit.append(noise_model.on_each(*qubits))
    elif noise_type == "mixed":
        noise_channels = [
            cirq.depolarize(p=noise_level / 4),
            cirq.amplitude_damp(gamma=noise_level / 4),
            cirq.phase_damp(gamma=noise_level / 4),
            cirq.bit_flip(p=noise_level / 4),
        ]
        for moment in circuit:
            noisy_circuit.append(moment)
            for channel in noise_channels:
                noisy_circuit.append(channel.on_each(*qubits))
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    return noisy_circuit


# =============================================================================
# Float64 Dataset Generation
# =============================================================================


def generate_5qubit_dataset_float64(
    output_dir="dataset_5qubit_float64",
    num_samples_per_cell=5000,
    num_qubits=5,
    min_depth=6,
    max_depth=9,
    noise_types=None,
    noise_levels=None,
    chunk_size=1000,
    seed=42,
):
    """
    Generate 5-qubit dataset in float64 precision.

    Key difference: Uses complex128 simulator and saves as float64.
    """
    os.makedirs(output_dir, exist_ok=True)

    if noise_types is None:
        noise_types = [
            "depolarizing",
            "amplitude_damping",
            "phase_damping",
            "bitflip",
            "mixed",
        ]
    if noise_levels is None:
        noise_levels = [0.05, 0.10, 0.15, 0.20]

    np.random.seed(seed)
    random.seed(seed)

    # CRITICAL: Use complex128 for full precision
    simulator = cirq.DensityMatrixSimulator(dtype=np.complex128)

    total_samples = num_samples_per_cell * len(noise_types) * len(noise_levels)

    print(f"Generating {total_samples} 5-qubit samples in float64...")
    print(f"Matrix size: 32x32")
    print(f"Simulator dtype: complex128")
    print(f"Output directory: {output_dir}")

    chunk_counter = 0
    X_chunk, Y_chunk, meta_chunk = [], [], []

    with tqdm(total=total_samples) as pbar:
        for noise_type in noise_types:
            for noise_level in noise_levels:
                for i in range(num_samples_per_cell):
                    depth = random.randint(min_depth, max_depth)
                    circuit = generate_random_circuit(num_qubits, depth)
                    qubits = sorted(circuit.all_qubits())

                    # Clean simulation (complex128)
                    rho_clean = simulator.simulate(
                        circuit, qubit_order=qubits
                    ).final_density_matrix

                    # Noisy simulation (complex128)
                    noisy_circuit = add_noise(circuit, noise_type, noise_level)
                    rho_noisy = simulator.simulate(
                        noisy_circuit, qubit_order=qubits
                    ).final_density_matrix

                    # Stack real/imag as float64 (shape: 32, 32, 2)
                    x_sample = np.stack(
                        [rho_noisy.real, rho_noisy.imag], axis=-1
                    ).astype(np.float64)
                    y_sample = np.stack(
                        [rho_clean.real, rho_clean.imag], axis=-1
                    ).astype(np.float64)

                    # Verify shape and dtype
                    assert x_sample.shape == (32, 32, 2), (
                        f"Bad X shape: {x_sample.shape}"
                    )
                    assert y_sample.shape == (32, 32, 2), (
                        f"Bad Y shape: {y_sample.shape}"
                    )
                    assert x_sample.dtype == np.float64, (
                        f"Bad X dtype: {x_sample.dtype}"
                    )
                    assert y_sample.dtype == np.float64, (
                        f"Bad Y dtype: {y_sample.dtype}"
                    )

                    X_chunk.append(x_sample)
                    Y_chunk.append(y_sample)
                    meta_chunk.append(
                        {
                            "noise_type": noise_type,
                            "noise_level": noise_level,
                            "depth": depth,
                        }
                    )

                    pbar.update(1)

                    # Save chunk when full
                    if len(X_chunk) >= chunk_size:
                        chunk_counter += 1
                        fname = f"{output_dir}/chunk_{chunk_counter:04d}.pt"
                        torch.save(
                            {
                                "X": torch.from_numpy(
                                    np.stack(X_chunk, axis=0)
                                ),  # float64
                                "Y": torch.from_numpy(
                                    np.stack(Y_chunk, axis=0)
                                ),  # float64
                                "meta": meta_chunk,
                            },
                            fname,
                        )
                        print(
                            f"Saved {fname} ({len(X_chunk)} samples, dtype={torch.from_numpy(X_chunk[0]).dtype})"
                        )
                        X_chunk, Y_chunk, meta_chunk = [], [], []

    # Save remainder
    if len(X_chunk) > 0:
        chunk_counter += 1
        fname = f"{output_dir}/chunk_{chunk_counter:04d}.pt"
        torch.save(
            {
                "X": torch.from_numpy(np.stack(X_chunk, axis=0)),
                "Y": torch.from_numpy(np.stack(Y_chunk, axis=0)),
                "meta": meta_chunk,
            },
            fname,
        )
        print(f"Saved final {fname} ({len(X_chunk)} samples)")

    print(f"\nDataset generation complete!")
    print(f"Total samples: {total_samples}")
    print(f"Total chunks: {chunk_counter}")
    print(f"Output: {output_dir}/")

    # Verification: load first chunk and check dtype
    verify_fname = f"{output_dir}/chunk_0001.pt"
    if os.path.exists(verify_fname):
        data = torch.load(verify_fname, weights_only=False)
        print(f"\nVerification:")
        print(f"  X dtype: {data['X'].dtype} (expected torch.float64)")
        print(f"  Y dtype: {data['Y'].dtype} (expected torch.float64)")
        print(f"  X shape: {data['X'].shape}")
        assert data["X"].dtype == torch.float64, "X is not float64!"
        assert data["Y"].dtype == torch.float64, "Y is not float64!"
        print("  OK - float64 verified!")


if __name__ == "__main__":
    generate_5qubit_dataset_float64(
        output_dir="dataset_5qubit_float64",
        num_samples_per_cell=5000,  # 5000 * 5 noise types * 4 levels = 100,000
        chunk_size=1000,
        seed=42,
    )
