#!/usr/bin/env python3
"""Generate missing 8-qubit chunks for cells 16-19 (mixed noise type)."""

import os
import cirq
import numpy as np
import random
import torch


def generate_circuit_8qubit(num_qubits=8, depth=None):
    if depth is None:
        depth = random.randint(6, 9)
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
    two_qubit_gates = [cirq.CZ, cirq.CNOT, cirq.SWAP]
    qubit_pairs = list(zip(qubits[:-1], qubits[1:]))

    for d in range(depth):
        for q in qubits:
            gate = random.choice(single_qubit_gates)
            circuit.append(gate(q))
        n_two_qubit = min(len(qubit_pairs), num_qubits // 2)
        for q1, q2 in random.sample(qubit_pairs, k=n_two_qubit):
            gate = random.choice(two_qubit_gates)
            circuit.append(gate(q1, q2))
    return circuit, qubits


def add_mixed_noise(circuit, noise_level):
    qubits = sorted(circuit.all_qubits())
    noisy_circuit = cirq.Circuit()
    for moment in circuit:
        noisy_circuit.append(moment)
        noisy_circuit.append(cirq.depolarize(p=noise_level / 4).on_each(*qubits))
        noisy_circuit.append(
            cirq.amplitude_damp(gamma=noise_level / 4).on_each(*qubits)
        )
        noisy_circuit.append(cirq.phase_damp(gamma=noise_level / 4).on_each(*qubits))
        noisy_circuit.append(cirq.bit_flip(p=noise_level / 4).on_each(*qubits))
    return noisy_circuit


def generate_chunk(
    cell_id, chunk_num, noise_level, output_dir, num_samples=1000, seed_offset=0
):
    seed = 42 + cell_id * 10000 + chunk_num * 1000 + seed_offset
    np.random.seed(seed)
    random.seed(seed)

    simulator = cirq.DensityMatrixSimulator(dtype=np.complex128)
    samples = []

    for i in range(num_samples):
        circuit, qubits = generate_circuit_8qubit()
        rho_clean = simulator.simulate(circuit, qubit_order=qubits).final_density_matrix
        noisy_circuit = add_mixed_noise(circuit, noise_level)
        rho_noisy = simulator.simulate(
            noisy_circuit, qubit_order=qubits
        ).final_density_matrix

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
                "meta": {"noise_type": "mixed", "noise_level": noise_level},
            }
        )

        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples")

    X_batch = np.stack([s["X"] for s in samples], axis=0)
    Y_batch = np.stack([s["Y"] for s in samples], axis=0)
    meta_batch = [s["meta"] for s in samples]

    fname = f"{output_dir}/cell{cell_id:02d}_chunk{chunk_num:02d}.pt"
    torch.save(
        {
            "X": torch.from_numpy(X_batch),
            "Y": torch.from_numpy(Y_batch),
            "meta": meta_batch,
        },
        fname,
    )
    print(f"Saved {fname}")


if __name__ == "__main__":
    # Missing chunks:
    # cell16 (mixed, 0.05): missing chunk 5
    # cell17 (mixed, 0.10): missing chunk 5
    # cell18 (mixed, 0.15): missing chunks 4, 5
    # cell19 (mixed, 0.20): missing chunks 4, 5

    missing = [
        (16, 5, 0.05),
        (17, 5, 0.10),
        (18, 4, 0.15),
        (18, 5, 0.15),
        (19, 4, 0.20),
        (19, 5, 0.20),
    ]

    output_dir = "dataset_8qubit_float64"

    print(f"Generating {len(missing)} missing chunks (6000 samples)...")
    for cell_id, chunk_num, noise_level in missing:
        print(
            f"\nGenerating cell{cell_id:02d}_chunk{chunk_num:02d} (mixed, level={noise_level})"
        )
        generate_chunk(cell_id, chunk_num, noise_level, output_dir, seed_offset=99999)
    print("\nDone! All 100 chunks complete.")
