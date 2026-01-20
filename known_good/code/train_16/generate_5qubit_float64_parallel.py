#!/usr/bin/env python3
"""
Parallel version of 5-qubit float64 dataset generation.

Uses multiprocessing to generate samples in parallel.
Each worker handles a subset of noise_type/noise_level cells.

Expected speedup: ~8x on 8-core machine (limited by GIL-free numpy/cirq operations)
"""

import os
import cirq
import numpy as np
import random
import torch
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm


def generate_random_circuit(num_qubits, depth):
    """Generate a random quantum circuit with specified number of qubits and depth."""
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
        for q in qubits:
            gate = random.choice(single_qubit_gates)
            circuit.append(gate(q))

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


def generate_cell(args):
    """
    Generate all samples for one (noise_type, noise_level) cell.
    This function runs in a worker process.

    Returns: (cell_id, samples_list)
    """
    (
        cell_id,
        noise_type,
        noise_level,
        num_samples,
        num_qubits,
        min_depth,
        max_depth,
        base_seed,
    ) = args

    # Each cell gets a unique seed based on cell_id
    seed = base_seed + cell_id * 10000
    np.random.seed(seed)
    random.seed(seed)

    simulator = cirq.DensityMatrixSimulator(dtype=np.complex128)

    samples = []
    for i in range(num_samples):
        depth = random.randint(min_depth, max_depth)
        circuit = generate_random_circuit(num_qubits, depth)
        qubits = sorted(circuit.all_qubits())

        # Clean simulation
        rho_clean = simulator.simulate(circuit, qubit_order=qubits).final_density_matrix

        # Noisy simulation
        noisy_circuit = add_noise(circuit, noise_type, noise_level)
        rho_noisy = simulator.simulate(
            noisy_circuit, qubit_order=qubits
        ).final_density_matrix

        # Stack real/imag as float64
        x_sample = np.stack([rho_noisy.real, rho_noisy.imag], axis=-1).astype(
            np.float64
        )
        y_sample = np.stack([rho_clean.real, rho_clean.imag], axis=-1).astype(
            np.float64
        )

        samples.append(
            {
                "X": x_sample,
                "Y": y_sample,
                "meta": {
                    "noise_type": noise_type,
                    "noise_level": noise_level,
                    "depth": depth,
                },
            }
        )

    return (cell_id, samples)


def generate_5qubit_dataset_parallel(
    output_dir="dataset_5qubit_float64",
    num_samples_per_cell=5000,
    num_qubits=5,
    min_depth=6,
    max_depth=9,
    noise_types=None,
    noise_levels=None,
    chunk_size=1000,
    seed=42,
    num_workers=None,
):
    """
    Generate 5-qubit dataset in float64 precision using parallel workers.

    Each (noise_type, noise_level) cell is processed by a separate worker.
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

    if num_workers is None:
        num_workers = min(cpu_count(), len(noise_types) * len(noise_levels))

    total_samples = num_samples_per_cell * len(noise_types) * len(noise_levels)
    num_cells = len(noise_types) * len(noise_levels)

    print(f"Generating {total_samples} 5-qubit samples in float64...")
    print(f"Matrix size: 32x32")
    print(f"Cells: {num_cells} (5 noise types x 4 levels)")
    print(f"Workers: {num_workers}")
    print(f"Output directory: {output_dir}")
    print()

    # Build list of cells to process
    cells = []
    cell_id = 0
    for noise_type in noise_types:
        for noise_level in noise_levels:
            cells.append(
                (
                    cell_id,
                    noise_type,
                    noise_level,
                    num_samples_per_cell,
                    num_qubits,
                    min_depth,
                    max_depth,
                    seed,
                )
            )
            cell_id += 1

    # Process cells in parallel with progress bar
    all_samples = []

    with Pool(num_workers) as pool:
        # imap_unordered for better progress tracking
        results = pool.imap_unordered(generate_cell, cells)

        with tqdm(total=num_cells, desc="Cells completed") as pbar:
            for cell_id, samples in results:
                all_samples.extend(samples)
                pbar.update(1)
                pbar.set_postfix({"samples": len(all_samples)})

    print(f"\nGenerated {len(all_samples)} samples, saving to chunks...")

    # Save in chunks
    chunk_counter = 0
    for i in range(0, len(all_samples), chunk_size):
        chunk_samples = all_samples[i : i + chunk_size]
        chunk_counter += 1

        X_batch = np.stack([s["X"] for s in chunk_samples], axis=0)
        Y_batch = np.stack([s["Y"] for s in chunk_samples], axis=0)
        meta_batch = [s["meta"] for s in chunk_samples]

        fname = f"{output_dir}/chunk_{chunk_counter:04d}.pt"
        torch.save(
            {
                "X": torch.from_numpy(X_batch),
                "Y": torch.from_numpy(Y_batch),
                "meta": meta_batch,
            },
            fname,
        )

    print(f"\nDataset generation complete!")
    print(f"Total samples: {len(all_samples)}")
    print(f"Total chunks: {chunk_counter}")
    print(f"Output: {output_dir}/")

    # Verification
    verify_fname = f"{output_dir}/chunk_0001.pt"
    if os.path.exists(verify_fname):
        data = torch.load(verify_fname, weights_only=False)
        print(f"\nVerification:")
        print(f"  X dtype: {data['X'].dtype} (expected torch.float64)")
        print(f"  Y dtype: {data['Y'].dtype} (expected torch.float64)")
        print(f"  X shape: {data['X'].shape}")
        assert data["X"].dtype == torch.float64, "X is not float64!"
        assert data["Y"].dtype == torch.float64, "Y is not float64!"
        print("  ✓ float64 verified!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="dataset_5qubit_float64", help="Output directory"
    )
    parser.add_argument(
        "--samples-per-cell", type=int, default=5000, help="Samples per noise cell"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of workers (default: cpu_count)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate_5qubit_dataset_parallel(
        output_dir=args.output,
        num_samples_per_cell=args.samples_per_cell,
        chunk_size=1000,
        seed=args.seed,
        num_workers=args.workers,
    )
