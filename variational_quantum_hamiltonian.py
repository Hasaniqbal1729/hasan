"""
Variational Quantum Hamiltonian Engineering (VQHE)

Implements:
  1. Pauli operators & Hamiltonian construction
  2. Parametrized quantum circuits (ansatz)
  3. Expectation value computation via statevector simulation
  4. Variational optimization (VQE-style)
  5. Hamiltonian engineering: match a target Hamiltonian via H_eff(θ) = U†(θ) H₀ U(θ)
"""

import numpy as np
from scipy.optimize import minimize
from itertools import product
from typing import List, Tuple, Dict, Optional

# ---------------------------------------------------------------------------
# Pauli matrices
# ---------------------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
X  = np.array([[0, 1], [1, 0]], dtype=complex)
Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z  = np.array([[1, 0], [0, -1]], dtype=complex)

PAULIS: Dict[str, np.ndarray] = {"I": I2, "X": X, "Y": Y, "Z": Z}


# ---------------------------------------------------------------------------
# Helper: tensor products
# ---------------------------------------------------------------------------

def kron_chain(ops: List[np.ndarray]) -> np.ndarray:
    """Tensor-product a list of 2×2 matrices into a 2^n × 2^n matrix."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def pauli_string_matrix(label: str) -> np.ndarray:
    """Convert a Pauli string like 'XZI' to the corresponding matrix."""
    return kron_chain([PAULIS[c] for c in label])


# ---------------------------------------------------------------------------
# Hamiltonian
# ---------------------------------------------------------------------------

class Hamiltonian:
    """
    H = Σ_i  c_i  P_i
    where P_i is a tensor-product of Pauli operators.
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.terms: List[Tuple[complex, str]] = []   # (coeff, pauli_string)

    def add_term(self, coeff: complex, pauli_string: str) -> "Hamiltonian":
        assert len(pauli_string) == self.n_qubits, (
            f"Pauli string length {len(pauli_string)} != n_qubits {self.n_qubits}"
        )
        self.terms.append((coeff, pauli_string))
        return self

    def matrix(self) -> np.ndarray:
        dim = 2 ** self.n_qubits
        H = np.zeros((dim, dim), dtype=complex)
        for coeff, ps in self.terms:
            H += coeff * pauli_string_matrix(ps)
        return H

    def expectation(self, state: np.ndarray) -> float:
        """⟨ψ|H|ψ⟩  (state is a complex column vector)."""
        Hv = self.matrix() @ state
        return float(np.real(state.conj() @ Hv))

    def __repr__(self):
        parts = [f"({c:.3f}){ps}" for c, ps in self.terms]
        return "H = " + " + ".join(parts)


# ---------------------------------------------------------------------------
# Pre-built Hamiltonians
# ---------------------------------------------------------------------------

def transverse_ising(n: int, J: float = 1.0, h: float = 0.5) -> Hamiltonian:
    """H = -J Σ ZZ  -  h Σ X   (periodic boundary)"""
    H = Hamiltonian(n)
    for i in range(n):
        ps = "I" * i + "Z" + "Z" + "I" * (n - i - 2)  # ZZ on (i, i+1)
        if i < n - 1:
            H.add_term(-J, ps)
    # Wrap-around
    ps_wrap = "Z" + "I" * (n - 2) + "Z"
    H.add_term(-J, ps_wrap)
    for i in range(n):
        ps = "I" * i + "X" + "I" * (n - i - 1)
        H.add_term(-h, ps)
    return H


def heisenberg(n: int, Jx: float = 1.0, Jy: float = 1.0, Jz: float = 1.0) -> Hamiltonian:
    """H = Σ (Jx XX + Jy YY + Jz ZZ) on nearest neighbours."""
    H = Hamiltonian(n)
    for i in range(n - 1):
        for coeff, char in [(Jx, "X"), (Jy, "Y"), (Jz, "Z")]:
            ps = "I" * i + char + char + "I" * (n - i - 2)
            H.add_term(coeff, ps)
    return H


# ---------------------------------------------------------------------------
# Parametrized quantum circuit (ansatz)
# ---------------------------------------------------------------------------

def _ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-1j * theta / 2), 0],
                     [0,  np.exp(1j * theta / 2)]], dtype=complex)


def _cnot(n: int, control: int, target: int) -> np.ndarray:
    dim = 2 ** n
    mat = np.eye(dim, dtype=complex)
    for row in range(dim):
        bits = format(row, f"0{n}b")
        if bits[control] == "1":
            new_bits = list(bits)
            new_bits[target] = "1" if bits[target] == "0" else "0"
            col = int("".join(new_bits), 2)
            mat[row, row] = 0
            mat[col, row] = 1
    return mat


class HardwareEfficientAnsatz:
    """
    Hardware-efficient ansatz: alternating layers of
      RY/RZ single-qubit rotations + linear CNOT entanglers.
    """

    def __init__(self, n_qubits: int, n_layers: int):
        self.n = n_qubits
        self.n_layers = n_layers
        # 2 params (RY, RZ) per qubit per layer  +  1 final RY layer
        self.n_params = 2 * n_qubits * n_layers + n_qubits

    def _entangler(self) -> np.ndarray:
        U = np.eye(2 ** self.n, dtype=complex)
        for i in range(self.n - 1):
            U = _cnot(self.n, i, i + 1) @ U
        return U

    def unitary(self, params: np.ndarray) -> np.ndarray:
        assert len(params) == self.n_params
        n = self.n
        U = np.eye(2 ** n, dtype=complex)
        idx = 0
        for _ in range(self.n_layers):
            # RY layer
            ry_layer = kron_chain([_ry(params[idx + q]) for q in range(n)])
            idx += n
            # RZ layer
            rz_layer = kron_chain([_rz(params[idx + q]) for q in range(n)])
            idx += n
            U = rz_layer @ ry_layer @ U
            U = self._entangler() @ U
        # Final RY
        ry_final = kron_chain([_ry(params[idx + q]) for q in range(n)])
        U = ry_final @ U
        return U

    def state(self, params: np.ndarray) -> np.ndarray:
        """Apply ansatz to |0…0⟩."""
        psi0 = np.zeros(2 ** self.n, dtype=complex)
        psi0[0] = 1.0
        return self.unitary(params) @ psi0


# ---------------------------------------------------------------------------
# VQE: Variational Quantum Eigensolver
# ---------------------------------------------------------------------------

class VQE:
    """
    Minimize ⟨ψ(θ)|H|ψ(θ)⟩ to approximate the ground-state energy.
    """

    def __init__(self, hamiltonian: Hamiltonian, ansatz: HardwareEfficientAnsatz):
        self.H = hamiltonian
        self.ansatz = ansatz
        self.history: List[float] = []

    def cost(self, params: np.ndarray) -> float:
        psi = self.ansatz.state(params)
        return self.H.expectation(psi)

    def run(self, init_params: Optional[np.ndarray] = None,
            method: str = "COBYLA", max_iter: int = 1000) -> dict:
        if init_params is None:
            rng = np.random.default_rng(42)
            init_params = rng.uniform(0, 2 * np.pi, self.ansatz.n_params)

        self.history = []

        def callback(p):
            self.history.append(self.cost(p))

        result = minimize(
            self.cost, init_params, method=method,
            callback=callback,
            options={"maxiter": max_iter, "rhobeg": 0.5}
        )

        # Exact ground-state energy for comparison
        evals = np.linalg.eigvalsh(self.H.matrix())
        exact = float(evals[0])

        return {
            "vqe_energy": result.fun,
            "exact_energy": exact,
            "error": abs(result.fun - exact),
            "n_iters": len(self.history),
            "optimal_params": result.x,
            "converged": result.success,
        }


# ---------------------------------------------------------------------------
# Hamiltonian Engineering: H_eff(θ) = U†(θ) H₀ U(θ)  →  H_target
# ---------------------------------------------------------------------------

class VariationalHamiltonianEngineering:
    """
    Find circuit parameters θ such that
        H_eff(θ) = U†(θ) H₀ U(θ)
    is as close as possible (in Frobenius norm) to H_target.

    This is useful for quantum simulation: apply a unitary frame change
    to map a native hardware Hamiltonian H₀ onto a desired target.
    """

    def __init__(self, H0: Hamiltonian, H_target: Hamiltonian,
                 ansatz: HardwareEfficientAnsatz):
        assert H0.n_qubits == H_target.n_qubits == ansatz.n
        self.H0_mat = H0.matrix()
        self.Htgt_mat = H_target.matrix()
        self.ansatz = ansatz
        self.history: List[float] = []

    def _h_eff(self, params: np.ndarray) -> np.ndarray:
        U = self.ansatz.unitary(params)
        return U.conj().T @ self.H0_mat @ U

    def cost(self, params: np.ndarray) -> float:
        diff = self._h_eff(params) - self.Htgt_mat
        return float(np.real(np.trace(diff.conj().T @ diff)))   # ||·||_F²

    def run(self, init_params: Optional[np.ndarray] = None,
            method: str = "L-BFGS-B", max_iter: int = 2000) -> dict:
        if init_params is None:
            rng = np.random.default_rng(0)
            init_params = rng.uniform(0, 2 * np.pi, self.ansatz.n_params)

        self.history = []

        def callback(p):
            self.history.append(self.cost(p))

        result = minimize(
            self.cost, init_params, method=method,
            callback=callback,
            options={"maxiter": max_iter}
        )

        opt_U = self.ansatz.unitary(result.x)
        H_eff_final = opt_U.conj().T @ self.H0_mat @ opt_U

        return {
            "frobenius_cost": result.fun,
            "n_iters": len(self.history),
            "optimal_params": result.x,
            "H_eff_matrix": H_eff_final,
            "H_target_matrix": self.Htgt_mat,
            "converged": result.success,
        }


# ---------------------------------------------------------------------------
# Pauli decomposition utility
# ---------------------------------------------------------------------------

def pauli_decompose(matrix: np.ndarray, n_qubits: int,
                    tol: float = 1e-8) -> List[Tuple[complex, str]]:
    """
    Decompose a 2^n × 2^n Hermitian matrix into a sum of Pauli strings.
    Returns a list of (coefficient, pauli_string) tuples (drops near-zero terms).
    """
    dim = 2 ** n_qubits
    assert matrix.shape == (dim, dim)
    terms = []
    labels = ["I", "X", "Y", "Z"]
    for combo in product(labels, repeat=n_qubits):
        ps = "".join(combo)
        P = pauli_string_matrix(ps)
        coeff = np.trace(P @ matrix) / dim
        if abs(coeff) > tol:
            terms.append((coeff, ps))
    return terms


# ---------------------------------------------------------------------------
# Demo / main
# ---------------------------------------------------------------------------

def print_section(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def demo_vqe():
    print_section("1. VQE — Transverse-Field Ising Model (3 qubits)")
    n = 3
    H = transverse_ising(n, J=1.0, h=0.8)
    ansatz = HardwareEfficientAnsatz(n_qubits=n, n_layers=2)
    vqe = VQE(H, ansatz)
    res = vqe.run(method="COBYLA", max_iter=2000)

    print(f"  Hamiltonian : {H}")
    print(f"  Ansatz      : {n} qubits, 2 layers, {ansatz.n_params} params")
    print(f"  VQE energy  : {res['vqe_energy']:.6f}")
    print(f"  Exact energy: {res['exact_energy']:.6f}")
    print(f"  |Error|     : {res['error']:.2e}")
    print(f"  Iterations  : {res['n_iters']}")
    return res


def demo_heisenberg_vqe():
    print_section("2. VQE — Heisenberg Chain (4 qubits)")
    n = 4
    H = heisenberg(n, Jx=1.0, Jy=1.0, Jz=0.5)
    ansatz = HardwareEfficientAnsatz(n_qubits=n, n_layers=3)
    vqe = VQE(H, ansatz)
    res = vqe.run(method="COBYLA", max_iter=3000)

    print(f"  Ansatz params : {ansatz.n_params}")
    print(f"  VQE energy    : {res['vqe_energy']:.6f}")
    print(f"  Exact energy  : {res['exact_energy']:.6f}")
    print(f"  |Error|       : {res['error']:.2e}")
    print(f"  Iterations    : {res['n_iters']}")
    return res


def demo_hamiltonian_engineering():
    print_section("3. Variational Hamiltonian Engineering (2 qubits)")
    n = 2

    # Native (hardware) Hamiltonian
    H0 = Hamiltonian(n)
    H0.add_term(1.0, "ZZ")
    H0.add_term(0.5, "XI")
    H0.add_term(0.5, "IX")

    # Target Hamiltonian (e.g., Heisenberg-like)
    H_target = Hamiltonian(n)
    H_target.add_term(1.0, "XX")
    H_target.add_term(1.0, "YY")
    H_target.add_term(1.0, "ZZ")

    ansatz = HardwareEfficientAnsatz(n_qubits=n, n_layers=3)
    eng = VariationalHamiltonianEngineering(H0, H_target, ansatz)
    res = eng.run(method="L-BFGS-B", max_iter=3000)

    print(f"  H₀      : {H0}")
    print(f"  H_target: {H_target}")
    print(f"  ||H_eff - H_target||²_F : {res['frobenius_cost']:.4e}")
    print(f"  Iterations              : {res['n_iters']}")
    print(f"  Converged               : {res['converged']}")

    print("\n  H_target (matrix):")
    _pretty_matrix(res["H_target_matrix"])
    print("\n  H_eff(θ*) (matrix):")
    _pretty_matrix(res["H_eff_matrix"])

    print("\n  Pauli decomposition of H_eff(θ*):")
    terms = pauli_decompose(res["H_eff_matrix"], n)
    for coeff, ps in sorted(terms, key=lambda t: -abs(t[0])):
        print(f"    {coeff.real:+.4f}  {ps}")

    return res


def demo_pauli_decompose():
    print_section("4. Pauli Decomposition Example (2 qubits)")
    # Build a known matrix and recover its Pauli expansion
    H = Hamiltonian(2)
    H.add_term(0.7, "ZZ")
    H.add_term(-0.3, "XX")
    H.add_term(0.5, "ZI")
    mat = H.matrix()

    terms = pauli_decompose(mat, n_qubits=2)
    print("  Original terms:", H)
    print("  Recovered terms:")
    for c, ps in terms:
        print(f"    {c.real:+.4f}  {ps}")


def _pretty_matrix(M: np.ndarray, decimals: int = 3):
    rows, cols = M.shape
    for r in range(rows):
        row_str = "  ["
        for c in range(cols):
            val = M[r, c]
            re, im = round(val.real, decimals), round(val.imag, decimals)
            if abs(im) < 1e-6:
                row_str += f" {re:+.3f}     "
            else:
                row_str += f" {re:+.3f}{im:+.3f}j "
        row_str += "]"
        print(row_str)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_vqe()
    demo_heisenberg_vqe()
    demo_hamiltonian_engineering()
    demo_pauli_decompose()
    print("\nDone.")
