"""
Pauli basis representation for density matrices.

Converts between standard matrix representation (real/imag flattened)
and Pauli basis coefficients.

For an n-qubit density matrix ρ:
    ρ = (1/2^n) Σ_i c_i P_i
where P_i are Pauli operators and c_i are real coefficients.

For n=5 qubits: 4^5 = 1024 Pauli operators.

Key insight: Pauli basis is a basis-invariant representation that might be
more suitable for MLPs than element-wise (spatial) representation.
"""

import numpy as np
import torch
import itertools


def get_pauli_string(index, n_qubits):
    """
    Get the i-th Pauli string for n qubits.
    Pauli strings indexed as: (I, X, Y, Z)^n in base-4 representation.

    Args:
        index: int, 0 <= index < 4^n
        n_qubits: int

    Returns:
        tuple of length n_qubits, each element in {0:I, 1:X, 2:Y, 3:Z}
    """
    pauli_tuple = []
    for _ in range(n_qubits):
        pauli_tuple.append(index % 4)
        index //= 4
    return tuple(reversed(pauli_tuple))


def pauli_matrix(pauli_char):
    """Get single-qubit Pauli matrix.

    Args:
        pauli_char: int in {0:I, 1:X, 2:Y, 3:Z}

    Returns:
        2x2 complex numpy array
    """
    if pauli_char == 0:  # I
        return np.array([[1, 0], [0, 1]], dtype=complex)
    elif pauli_char == 1:  # X
        return np.array([[0, 1], [1, 0]], dtype=complex)
    elif pauli_char == 2:  # Y
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    else:  # Z (3)
        return np.array([[1, 0], [0, -1]], dtype=complex)


def pauli_string_matrix(pauli_string):
    """Get full Pauli string matrix via tensor product.

    Args:
        pauli_string: tuple of ints, each in {0:I, 1:X, 2:Y, 3:Z}

    Returns:
        2^n x 2^n complex numpy array
    """
    mat = pauli_matrix(pauli_string[0])
    for pauli_char in pauli_string[1:]:
        mat = np.kron(mat, pauli_matrix(pauli_char))
    return mat


def density_matrix_to_pauli_basis(rho_matrix, n_qubits=5):
    """
    Convert density matrix to Pauli basis coefficients.

    Args:
        rho_matrix: (2^n, 2^n) complex array or (2, 2^n, 2^n) tensor (real/imag parts)
        n_qubits: int

    Returns:
        (4^n,) real array of Pauli expectation values
    """
    # Handle (2, 32, 32) format: real and imag parts
    if isinstance(rho_matrix, torch.Tensor):
        rho_matrix = rho_matrix.cpu().numpy()

    if rho_matrix.shape == (2, 2**n_qubits, 2**n_qubits):
        rho = rho_matrix[0] + 1j * rho_matrix[1]
    else:
        rho = rho_matrix

    dim = 2**n_qubits
    pauli_coeffs = np.zeros(4**n_qubits)

    for i in range(4**n_qubits):
        pauli_string = get_pauli_string(i, n_qubits)
        P_i = pauli_string_matrix(pauli_string)
        c_i = np.real(np.trace(rho @ P_i))  # Tr(ρ P_i)
        pauli_coeffs[i] = c_i

    return pauli_coeffs


def pauli_basis_to_density_matrix(pauli_coeffs, n_qubits=5):
    """
    Reconstruct density matrix from Pauli basis coefficients.

    ρ = (1/2^n) Σ_i c_i P_i

    Args:
        pauli_coeffs: (4^n,) real array
        n_qubits: int

    Returns:
        (2^n, 2^n) complex array
    """
    dim = 2**n_qubits
    rho = np.zeros((dim, dim), dtype=complex)

    for i in range(4**n_qubits):
        pauli_string = get_pauli_string(i, n_qubits)
        P_i = pauli_string_matrix(pauli_string)
        rho += pauli_coeffs[i] * P_i

    rho /= dim
    return rho


class PauliRepresentationDataset:
    """Wrapper to convert dataset on-the-fly to Pauli basis."""

    def __init__(self, X, Y, n_qubits=5):
        """
        Args:
            X: (N, 2, 32, 32) noisy density matrices
            Y: (N, 2, 32, 32) clean density matrices
            n_qubits: int
        """
        self.X = X
        self.Y = Y
        self.n_qubits = n_qubits
        self.length = len(X)

        # Precompute Pauli basis representations for efficiency
        print("Converting dataset to Pauli basis...")
        self.X_pauli = np.array([
            density_matrix_to_pauli_basis(X[i], n_qubits)
            for i in range(len(X))
        ])
        self.Y_pauli = np.array([
            density_matrix_to_pauli_basis(Y[i], n_qubits)
            for i in range(len(Y))
        ])
        print(f"  X_pauli shape: {self.X_pauli.shape}")
        print(f"  Y_pauli shape: {self.Y_pauli.shape}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Returns Pauli basis representation as (4^n,) arrays."""
        return self.X_pauli[idx], self.Y_pauli[idx]


if __name__ == "__main__":
    # Test: convert small density matrix to Pauli basis and back
    n_qubits = 2

    # Create random density matrix
    A = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
    rho = A @ A.conj().T
    rho /= np.trace(rho)

    print(f"Original density matrix:\n{rho}\n")

    # Convert to Pauli basis
    pauli_coeffs = density_matrix_to_pauli_basis(rho, n_qubits)
    print(f"Pauli coefficients: {pauli_coeffs}\n")

    # Reconstruct
    rho_reconstructed = pauli_basis_to_density_matrix(pauli_coeffs, n_qubits)
    print(f"Reconstructed density matrix:\n{rho_reconstructed}\n")

    # Check reconstruction error
    error = np.linalg.norm(rho - rho_reconstructed, 'fro')
    print(f"Frobenius error: {error:.2e}")
