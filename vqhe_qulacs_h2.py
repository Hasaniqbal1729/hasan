"""
Variational Quantum Hamiltonian Engineering (VQHE) — Qulacs Implementation
===========================================================================
Paper : "Variational Quantum Hamiltonian Engineering"
        Benchi Zhao, Keisuke Fujii
        arXiv:2406.08998 | Phys. Rev. Research 7, 023123 (2025)

Goal
----
Minimise the Pauli norm  ||H'||_p = Σ_P |h'_P|  of the *engineered*
Hamiltonian  H'(θ) = U†(θ) H U(θ),  where U(θ) is a parametric quantum
circuit (PQC).  A smaller Pauli norm directly reduces measurement overhead
(∝ ||H||_p²) and Hamiltonian-simulation gate count (∝ ||H||_p).

H₂ Hamiltonian (STO-3G, R = 0.7414 Å, Jordan-Wigner, 2-qubit reduced)
-----------------------------------------------------------------------
H = g₀ II + g₁ ZI + g₂ IZ + g₃ ZZ + g₄ XX + g₅ YY
  = −1.0523732 II + 0.3979374 ZI − 0.3979374 IZ
    − 0.0112801 ZZ + 0.1809312 XX + 0.1809312 YY

Requirements: pip install qulacs numpy scipy
"""

import numpy as np
from scipy.optimize import minimize
from itertools import product as iproduct
from typing import Dict, List, Tuple

from qulacs import QuantumState, ParametricQuantumCircuit, Observable

# ═══════════════════════════════════════════════════════════════
# §1  Pauli algebra
# ═══════════════════════════════════════════════════════════════

_P = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def pauli_matrix(label: str) -> np.ndarray:
    out = _P[label[0]]
    for c in label[1:]:
        out = np.kron(out, _P[c])
    return out


def hamiltonian_matrix(coeffs: Dict[str, float]) -> np.ndarray:
    n = len(next(iter(coeffs)))
    H = np.zeros((2**n, 2**n), dtype=complex)
    for label, c in coeffs.items():
        H += c * pauli_matrix(label)
    return H


def pauli_decompose(matrix: np.ndarray, n: int,
                    tol: float = 1e-10) -> Dict[str, float]:
    """h_P = Tr(P · M) / 2^n  for all n-qubit Pauli strings P."""
    dim = 2**n
    out: Dict[str, float] = {}
    for combo in iproduct("IXYZ", repeat=n):
        label = "".join(combo)
        c = float(np.real(np.trace(pauli_matrix(label) @ matrix))) / dim
        if abs(c) > tol:
            out[label] = c
    return out


def pauli_norm(coeffs: Dict[str, float]) -> float:
    return sum(abs(v) for v in coeffs.values())


# ═══════════════════════════════════════════════════════════════
# §2  H₂ molecular Hamiltonian (2-qubit reduced form)
# ═══════════════════════════════════════════════════════════════

H2_COEFFS: Dict[str, float] = {
    "II": -1.0523732,
    "ZI":  0.3979374,
    "IZ": -0.3979374,
    "ZZ": -0.0112801,
    "XX":  0.1809312,
    "YY":  0.1809312,
}
N_QUBITS = 2

# ═══════════════════════════════════════════════════════════════
# §3  Qulacs Observable  (qubit index convention fix)
# ═══════════════════════════════════════════════════════════════
#
# Qulacs uses LSB-first qubit ordering (qubit 0 = least significant bit),
# while numpy kron products use MSB-first (qubit 0 = most significant bit).
# Fix: reverse qubit indices  i  →  n-1-i  when building the Observable.

def build_observable(coeffs: Dict[str, float], n: int) -> Observable:
    obs = Observable(n)
    for label, coeff in coeffs.items():
        if abs(coeff) < 1e-12:
            continue
        # Reverse qubit index to convert MSB→LSB convention
        parts = [f"{p} {n-1-i}" for i, p in enumerate(label) if p != "I"]
        pauli_str = " ".join(parts) if parts else "I 0"
        obs.add_operator(float(coeff), pauli_str)
    return obs


def ev_observable(obs: Observable, psi: np.ndarray, n: int) -> float:
    """⟨ψ|O|ψ⟩ via Qulacs Observable (loads numpy vector directly)."""
    s = QuantumState(n)
    s.load(psi)
    return float(np.real(obs.get_expectation_value(s)))


def ev_matrix(psi: np.ndarray, H: np.ndarray) -> float:
    """⟨ψ|H|ψ⟩ via direct matrix multiply."""
    return float(np.real(psi.conj() @ H @ psi))


# ═══════════════════════════════════════════════════════════════
# §4  Hardware-efficient ansatz
# ═══════════════════════════════════════════════════════════════

class HEAnsatz:
    """
    L layers of [RY, RZ] per qubit + linear CNOT,  then a final RY layer.
    Parameters: 2·n·L + n  (all initialised to 0 → almost-identity).
    """

    def __init__(self, n_qubits: int, n_layers: int):
        self.n = n_qubits
        self.L = n_layers
        self.pqc = ParametricQuantumCircuit(n_qubits)
        self._build()

    def _build(self):
        for _ in range(self.L):
            for q in range(self.n):
                self.pqc.add_parametric_RY_gate(q, 0.0)
                self.pqc.add_parametric_RZ_gate(q, 0.0)
            for q in range(self.n - 1):
                self.pqc.add_CNOT_gate(q, q + 1)
        for q in range(self.n):
            self.pqc.add_parametric_RY_gate(q, 0.0)

    @property
    def n_params(self) -> int:
        return self.pqc.get_parameter_count()

    def set_params(self, params: np.ndarray):
        for k, v in enumerate(params):
            self.pqc.set_parameter(k, float(v))

    def unitary(self) -> np.ndarray:
        """Build full 2^n × 2^n unitary column-by-column."""
        dim = 2**self.n
        U = np.empty((dim, dim), dtype=complex)
        for j in range(dim):
            s = QuantumState(self.n)
            s.set_computational_basis(j)
            self.pqc.update_quantum_state(s)
            U[:, j] = s.get_vector()
        return U


# ═══════════════════════════════════════════════════════════════
# §5  Engineered Hamiltonian  H'(θ) = U†(θ) H U(θ)
# ═══════════════════════════════════════════════════════════════

def h_engineered(ansatz: HEAnsatz, H_mat: np.ndarray) -> np.ndarray:
    U = ansatz.unitary()
    return U.conj().T @ H_mat @ U


# ═══════════════════════════════════════════════════════════════
# §6  Cost function  C(θ) = ||H'(θ)||_p
# ═══════════════════════════════════════════════════════════════

def cost(params: np.ndarray, ansatz: HEAnsatz, H_mat: np.ndarray) -> float:
    ansatz.set_params(params)
    return pauli_norm(pauli_decompose(h_engineered(ansatz, H_mat), ansatz.n))


# ═══════════════════════════════════════════════════════════════
# §7  Gradient via parameter-shift rule  (correct Pauli-norm formula)
# ═══════════════════════════════════════════════════════════════
#
# ∂C/∂θ_k = Σ_P  sign(h'_P(θ)) × [h'_P(θ+π/2 ê_k) − h'_P(θ−π/2 ê_k)] / 2
#
# This is different from naively shifting the full cost C: the sign
# factor must be evaluated at the *current* point θ, not the shifted one.

def cost_gradient(params: np.ndarray,
                  ansatz: HEAnsatz,
                  H_mat: np.ndarray) -> np.ndarray:
    n = ansatz.n
    s = np.pi / 2

    # Current Pauli coefficients (used for sign)
    ansatz.set_params(params)
    c0 = pauli_decompose(h_engineered(ansatz, H_mat), n)

    grad = np.zeros(len(params))
    for k in range(len(params)):
        pp = params.copy(); pp[k] += s
        pm = params.copy(); pm[k] -= s

        ansatz.set_params(pp)
        cp = pauli_decompose(h_engineered(ansatz, H_mat), n)
        ansatz.set_params(pm)
        cm = pauli_decompose(h_engineered(ansatz, H_mat), n)

        all_p = set(c0) | set(cp) | set(cm)
        gk = 0.0
        for lbl in all_p:
            s0 = np.sign(c0.get(lbl, 0.0))
            if s0 != 0:
                gk += s0 * (cp.get(lbl, 0.0) - cm.get(lbl, 0.0)) / 2.0
        grad[k] = gk

    ansatz.set_params(params)   # restore
    return grad


# ═══════════════════════════════════════════════════════════════
# §8  VQHE optimisation  (multi-restart)
# ═══════════════════════════════════════════════════════════════

def run_vqhe(H_mat: np.ndarray,
             n_qubits: int,
             n_layers: int = 4,
             n_restarts: int = 6,
             max_iter: int = 600) -> dict:

    ansatz = HEAnsatz(n_qubits, n_layers)
    np_rng = np.random.default_rng(0)
    best_cost = np.inf
    best_params = None
    history_best: List[float] = []

    print(f"  Ansatz: {n_layers} layers, {ansatz.n_params} params, "
          f"{n_restarts} restarts")

    for restart in range(n_restarts):
        # Small-angle init keeps U close to a simple gate (not pure identity
        # due to CNOT layers, but a physically sensible starting point)
        theta0 = np_rng.uniform(-np.pi / 4, np.pi / 4, ansatz.n_params)
        history: List[float] = []

        def cb(p):
            history.append(cost(p, ansatz, H_mat))

        # Stage 1: gradient-free (COBYLA) – escapes flat regions
        res1 = minimize(
            cost, theta0,
            args=(ansatz, H_mat),
            method="COBYLA",
            callback=cb,
            options={"maxiter": 300, "rhobeg": 0.3},
        )
        # Stage 2: gradient-based (L-BFGS-B) – fine-tune
        res2 = minimize(
            cost, res1.x,
            args=(ansatz, H_mat),
            jac=cost_gradient,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "ftol": 1e-14, "gtol": 1e-9},
        )

        c_final = res2.fun
        print(f"    restart {restart+1}/{n_restarts}  "
              f"||H'||_p = {c_final:.6f}")

        if c_final < best_cost:
            best_cost = c_final
            best_params = res2.x.copy()
            history_best = history + [c_final]

    # Build final engineered Hamiltonian
    ansatz.set_params(best_params)
    H_eng = h_engineered(ansatz, H_mat)
    eng_coeffs = pauli_decompose(H_eng, n_qubits)

    return {
        "params":     best_params,
        "ansatz":     ansatz,
        "H_eng":      H_eng,
        "eng_coeffs": eng_coeffs,
        "history":    history_best,
        "final_cost": best_cost,
    }


# ═══════════════════════════════════════════════════════════════
# §9  Expectation-value comparison
# ═══════════════════════════════════════════════════════════════

def expectation_values(H_mat: np.ndarray, H_eng: np.ndarray,
                       H_obs: Observable, H_eng_obs: Observable,
                       n: int) -> dict:
    evals_H,  evecs_H  = np.linalg.eigh(H_mat)
    evals_Hp, evecs_Hp = np.linalg.eigh(H_eng)

    dim = 2**n
    psi0   = np.zeros(dim, dtype=complex); psi0[0] = 1.0
    psi_gs = evecs_H[:, 0]     # ground state of H
    psi_ep = evecs_Hp[:, 0]    # ground state of H'

    states = {
        "|0⟩":              psi0,
        "Ground state of H":  psi_gs,
        "Ground state of H'": psi_ep,
    }

    rows = {}
    for name, psi in states.items():
        rows[name] = {
            # Matrix path (reference)
            "evH_mat":  ev_matrix(psi, H_mat),
            "evHp_mat": ev_matrix(psi, H_eng),
            # Qulacs Observable path
            "evH_obs":  ev_observable(H_obs,     psi, n),
            "evHp_obs": ev_observable(H_eng_obs, psi, n),
        }

    return {
        "rows":    rows,
        "evals_H":  evals_H,
        "evals_Hp": evals_Hp,
    }


# ═══════════════════════════════════════════════════════════════
# §10  Display helpers
# ═══════════════════════════════════════════════════════════════

W = 62
SEP  = "─" * W
SEP2 = "═" * W


def print_pauli_table(coeffs: Dict[str, float], title: str):
    max_abs = max(abs(v) for v in coeffs.values())
    print(f"\n  {title}")
    print(f"  {'Pauli':^6}  {'Coefficient':>13}  {'|h_P|':>9}  Bar")
    print("  " + SEP)
    for lbl, c in sorted(coeffs.items(), key=lambda x: -abs(x[1])):
        bar = "█" * max(1, round(abs(c) / max_abs * 18))
        print(f"  {lbl:^6}  {c:+13.7f}  {abs(c):9.7f}  {bar}")
    pn = pauli_norm(coeffs)
    print("  " + SEP)
    print(f"  {'||H||_p':^6}  {'':13}  {pn:9.7f}")


def print_ev_table(rows: dict, label_H: str = "H", label_Hp: str = "H'"):
    hdr = (f"  {'State':<26}  "
           f"{'<'+label_H+'> mat':>13}  "
           f"{'<'+label_Hp+'> mat':>13}  "
           f"{'<'+label_H+'> Qulacs':>14}  "
           f"{'<'+label_Hp+'> Qulacs':>14}")
    print(hdr)
    print("  " + SEP)
    for name, v in rows.items():
        print(f"  {name:<26}  "
              f"{v['evH_mat']:+13.6f}  "
              f"{v['evHp_mat']:+13.6f}  "
              f"{v['evH_obs']:+14.6f}  "
              f"{v['evHp_obs']:+14.6f}")
    print("  " + SEP)


# ═══════════════════════════════════════════════════════════════
# §11  Main
# ═══════════════════════════════════════════════════════════════

def main():
    print(SEP2)
    print("  VQHE  |  arXiv:2406.08998  |  Simulator: Qulacs")
    print("  H₂  STO-3G  R=0.7414 Å  Jordan-Wigner  2 qubits")
    print(SEP2)

    n = N_QUBITS

    # ── Original Hamiltonian ───────────────────────────────────
    print("\n[1]  Original H₂ Hamiltonian")
    H_mat       = hamiltonian_matrix(H2_COEFFS)
    H_obs       = build_observable(H2_COEFFS, n)
    orig_coeffs = pauli_decompose(H_mat, n)
    orig_pn     = pauli_norm(orig_coeffs)

    print_pauli_table(orig_coeffs, "Pauli decomposition of H")

    evals_H = np.linalg.eigvalsh(H_mat)
    print(f"\n  Eigenvalues: " +
          "  ".join(f"{e:+.6f}" for e in evals_H) + " Ha")
    print(f"  Pauli norm  ||H||_p   = {orig_pn:.7f}")
    print(f"  Meas. overhead  ∝ ||H||_p²  = {orig_pn**2:.5f}")

    # Theoretical minimum: Pauli norm of diagonalised H
    D = np.diag(evals_H)
    min_pn = pauli_norm(pauli_decompose(D, n))
    print(f"  Theoretical minimum ||H||_p (diagonal form) = {min_pn:.7f}")

    # ── VQHE optimisation ──────────────────────────────────────
    print(f"\n[2]  VQHE Optimisation")
    res = run_vqhe(H_mat, n_qubits=n, n_layers=4,
                   n_restarts=6, max_iter=600)

    # ── Engineered Hamiltonian ─────────────────────────────────
    H_eng      = res["H_eng"]
    eng_coeffs = res["eng_coeffs"]
    eng_pn     = pauli_norm(eng_coeffs)
    H_eng_obs  = build_observable(eng_coeffs, n)

    print(f"\n[3]  Engineered Hamiltonian  H' = U†(θ*)HU(θ*)")
    print_pauli_table(eng_coeffs, "Pauli decomposition of H'")

    evals_Hp = np.linalg.eigvalsh(H_eng)
    print(f"\n  Eigenvalues: " +
          "  ".join(f"{e:+.6f}" for e in evals_Hp) + " Ha")

    reduction_norm = (1 - eng_pn / orig_pn) * 100
    reduction_ovhd = (1 - eng_pn**2 / orig_pn**2) * 100
    print(f"\n  Original  Pauli norm  ||H||_p   = {orig_pn:.7f}")
    print(f"  Engineered Pauli norm ||H'||_p  = {eng_pn:.7f}")
    print(f"  Theoretical minimum             = {min_pn:.7f}")
    print(f"  Norm reduction                  = {reduction_norm:+.2f}%")
    print(f"  Measurement overhead reduction  = {reduction_ovhd:+.2f}%")

    # ── Expectation values ─────────────────────────────────────
    print(f"\n[4]  Expectation Values   ⟨ψ|H|ψ⟩  and  ⟨ψ|H'|ψ⟩")
    ev = expectation_values(H_mat, H_eng, H_obs, H_eng_obs, n)
    print_ev_table(ev["rows"])

    gs_H  = float(ev["evals_H"][0])
    gs_Hp = float(ev["evals_Hp"][0])
    print(f"\n  Exact ground energy  E₀(H)  = {gs_H:.7f} Ha")
    print(f"  Exact ground energy  E₀(H') = {gs_Hp:.7f} Ha")
    print(f"  |ΔE₀|                       = {abs(gs_H-gs_Hp):.2e} Ha "
          "(must be ~0 — isospectrality)")

    # ── Verification ──────────────────────────────────────────
    print(f"\n[5]  Verification")
    max_eval_diff = np.max(np.abs(
        np.sort(ev["evals_H"]) - np.sort(ev["evals_Hp"])))
    print(f"  max |λ(H) − λ(H')|  = {max_eval_diff:.2e}  "
          + ("✓ isospectral" if max_eval_diff < 1e-7 else "⚠ NOT isospectral"))

    U = res["ansatz"].unitary()
    unitary_err = np.max(np.abs(U.conj().T @ U - np.eye(2**n)))
    print(f"  max |U†U − I|        = {unitary_err:.2e}  "
          + ("✓ unitary" if unitary_err < 1e-10 else "⚠ NOT unitary"))

    recon_err = np.max(np.abs(U.conj().T @ H_mat @ U - H_eng))
    print(f"  max |U†HU − H'|      = {recon_err:.2e}  "
          + ("✓ consistent" if recon_err < 1e-10 else "⚠ inconsistent"))

    # Matrix vs Qulacs agreement for |0⟩
    psi0 = np.zeros(2**n, dtype=complex); psi0[0] = 1.0
    diff_H  = abs(ev_matrix(psi0, H_mat) - ev_observable(H_obs, psi0, n))
    diff_Hp = abs(ev_matrix(psi0, H_eng) - ev_observable(H_eng_obs, psi0, n))
    print(f"  |⟨0|H|0⟩_mat − ⟨0|H|0⟩_Qulacs|  = {diff_H:.2e}  "
          + ("✓" if diff_H < 1e-8 else "⚠"))
    print(f"  |⟨0|H'|0⟩_mat − ⟨0|H'|0⟩_Qulacs| = {diff_Hp:.2e}  "
          + ("✓" if diff_Hp < 1e-8 else "⚠"))

    # ── Summary ────────────────────────────────────────────────
    print(f"\n{SEP2}")
    print("  SUMMARY")
    print(SEP2)
    print(f"  Original   Pauli norm  ||H||_p   = {orig_pn:.6f}")
    print(f"  Engineered Pauli norm  ||H'||_p  = {eng_pn:.6f}")
    print(f"  Theoretical minimum              = {min_pn:.6f}")
    print(f"  Norm reduction                   = {reduction_norm:+.2f}%")
    print(f"  Measurement overhead reduction   = {reduction_ovhd:+.2f}%")
    print(f"  Ansatz parameters                = {res['ansatz'].n_params}")
    print(f"  COBYLA+LBFGS restarts            = 6")
    print(SEP2)


if __name__ == "__main__":
    main()
