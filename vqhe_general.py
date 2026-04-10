"""
Variational Quantum Hamiltonian Engineering (VQHE)
===================================================
Paper : "Variational Quantum Hamiltonian Engineering"
        Benchi Zhao, Keisuke Fujii  |  arXiv:2406.08998

INPUT  → any n-qubit Hamiltonian as {pauli_string: coefficient}
OUTPUT → engineered Hamiltonian dict with minimised Pauli norm
         + expectation values of both H and H' for key states

Quick usage
-----------
    from vqhe_general import vqhe

    H = {"ZZ": 1.0, "XI": 0.5, "IX": 0.5}
    result = vqhe(H)               # run VQHE
    H_eng  = result["H_engineered"]

    # H_eng is the same dict format you can pass back in:
    vqhe(H_eng)   # norm should already be minimal

Requirements:  pip install qulacs numpy scipy
"""

import numpy as np
from scipy.optimize import minimize
from itertools import product as iproduct
from typing import Dict, List, Optional, Tuple

from qulacs import QuantumState, Observable

# ═══════════════════════════════════════════════════════════════
# §1  Pauli algebra  (vectorised, cached)
# ═══════════════════════════════════════════════════════════════

_P = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1],  [1, 0]],  dtype=complex),
    "Y": np.array([[0,-1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0],  [0, -1]], dtype=complex),
}

_BASIS_CACHE: Dict[int, Tuple[List[str], np.ndarray]] = {}


def _pauli_basis(n: int) -> Tuple[List[str], np.ndarray]:
    """Return (labels, matrices) for all 4^n n-qubit Pauli strings (cached)."""
    if n not in _BASIS_CACHE:
        labels, mats = [], []
        for combo in iproduct("IXYZ", repeat=n):
            lbl = "".join(combo)
            labels.append(lbl)
            M = _P[combo[0]]
            for c in combo[1:]:
                M = np.kron(M, _P[c])
            mats.append(M)
        _BASIS_CACHE[n] = (labels, np.array(mats, dtype=complex))
    return _BASIS_CACHE[n]


def coeffs_to_matrix(coeffs: Dict[str, float]) -> np.ndarray:
    """Pauli-coefficient dict → 2^n × 2^n Hermitian matrix."""
    n   = len(next(iter(coeffs)))
    lbl, mats = _pauli_basis(n)
    lbl_idx = {l: i for i, l in enumerate(lbl)}
    H = np.zeros((2**n, 2**n), dtype=complex)
    for l, c in coeffs.items():
        H += c * mats[lbl_idx[l]]
    return H


def matrix_to_coeffs(M: np.ndarray, n: int,
                     tol: float = 1e-9) -> Dict[str, float]:
    """
    2^n × 2^n matrix → Pauli-coefficient dict.
    Uses einsum for vectorised Tr(P_k M) over all k simultaneously.
    """
    labels, mats = _pauli_basis(n)
    dim = 2**n
    # Tr(P_k M) = sum_{ij} P_k[i,j] * M[j,i]  =  einsum('kij,ji->k', P, M)
    traces = np.einsum("kij,ji->k", mats, M).real / dim
    return {lbl: float(c)
            for lbl, c in zip(labels, traces) if abs(c) > tol}


def pauli_norm(coeffs: Dict[str, float]) -> float:
    """||H||_p = Σ_P |h_P|."""
    return float(sum(abs(v) for v in coeffs.values()))


def validate(coeffs: Dict[str, float]) -> int:
    """Validate Hamiltonian dict; return n_qubits."""
    lengths = {len(k) for k in coeffs}
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent Pauli-string lengths: {lengths}")
    n = lengths.pop()
    for lbl in coeffs:
        bad = set(lbl) - set("IXYZ")
        if bad:
            raise ValueError(f"Unknown characters {bad} in '{lbl}'")
    return n


# ═══════════════════════════════════════════════════════════════
# §2  Fast numpy ansatz  (no Qulacs overhead in optimisation loop)
# ═══════════════════════════════════════════════════════════════

class _Ansatz:
    """
    Hardware-efficient ansatz:
      L × [RY(θ) + RZ(φ) per qubit  +  linear CNOT]  +  final RY layer
    Unitary is built via sequential numpy matrix multiplication.
    CNOT matrices are precomputed and cached.
    """

    def __init__(self, n: int, layers: int):
        self.n = n
        self.L = layers
        # ── Build gate schedule ───────────────────────────────
        # Each entry: ("RY"|"RZ"|"CNOT", target_qubit, param_index_or_None)
        self._sched: List[Tuple] = []
        self.n_params = 0
        for _ in range(layers):
            for q in range(n):
                self._sched.append(("RY", q, self.n_params)); self.n_params += 1
                self._sched.append(("RZ", q, self.n_params)); self.n_params += 1
            for q in range(n - 1):
                self._sched.append(("CNOT", q, None))
        for q in range(n):
            self._sched.append(("RY", q, self.n_params)); self.n_params += 1

        # ── Precompute CNOT(q, q+1) matrices ─────────────────
        dim = 2**n
        self._cnot: Dict[int, np.ndarray] = {}
        for ctrl in range(n - 1):
            tgt = ctrl + 1
            mat = np.eye(dim, dtype=complex)
            for row in range(dim):
                bits = format(row, f"0{n}b")
                if bits[ctrl] == "1":
                    new = list(bits)
                    new[tgt] = "0" if bits[tgt] == "1" else "1"
                    col = int("".join(new), 2)
                    mat[row, row] = 0.0
                    mat[col, row] = 1.0
            self._cnot[ctrl] = mat

    # ── Gate helpers ─────────────────────────────────────────
    @staticmethod
    def _ry(t: float) -> np.ndarray:
        c, s = np.cos(t / 2), np.sin(t / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)

    @staticmethod
    def _rz(t: float) -> np.ndarray:
        e = np.exp(1j * t / 2)
        return np.array([[e.conj(), 0], [0, e]], dtype=complex)

    def _embed(self, G: np.ndarray, q: int) -> np.ndarray:
        """Embed a 2×2 gate G acting on qubit q into the n-qubit space."""
        ops = [np.eye(2, dtype=complex)] * self.n
        ops[q] = G
        out = ops[0]
        for op in ops[1:]:
            out = np.kron(out, op)
        return out

    # ── Main interface ────────────────────────────────────────
    def unitary(self, params: np.ndarray) -> np.ndarray:
        """Build the full 2^n × 2^n unitary U(params)."""
        dim = 2**self.n
        U = np.eye(dim, dtype=complex)
        for kind, q, p_idx in self._sched:
            if kind == "RY":
                G = self._embed(self._ry(params[p_idx]), q)
            elif kind == "RZ":
                G = self._embed(self._rz(params[p_idx]), q)
            else:                          # CNOT
                G = self._cnot[q]
            U = G @ U
        return U


# ═══════════════════════════════════════════════════════════════
# §3  Cost function  C(θ) = ||H'(θ)||_p  and its exact gradient
# ═══════════════════════════════════════════════════════════════

def _h_eng(U: np.ndarray, H_mat: np.ndarray) -> np.ndarray:
    return U.conj().T @ H_mat @ U


def _cost(params: np.ndarray,
          ansatz: _Ansatz,
          H_mat: np.ndarray) -> float:
    U = ansatz.unitary(params)
    return pauli_norm(matrix_to_coeffs(_h_eng(U, H_mat), ansatz.n))


def _gradient(params: np.ndarray,
              ansatz: _Ansatz,
              H_mat: np.ndarray) -> np.ndarray:
    """
    Exact gradient via parameter-shift rule:
      ∂C/∂θ_k = Σ_P  sign(h'_P(θ)) × [h'_P(θ+π/2) − h'_P(θ−π/2)] / 2
    """
    n  = ansatz.n
    sh = np.pi / 2
    c0 = matrix_to_coeffs(_h_eng(ansatz.unitary(params), H_mat), n)
    grad = np.zeros(len(params))

    for k in range(len(params)):
        pp = params.copy(); pp[k] += sh
        pm = params.copy(); pm[k] -= sh
        cp = matrix_to_coeffs(_h_eng(ansatz.unitary(pp), H_mat), n)
        cm = matrix_to_coeffs(_h_eng(ansatz.unitary(pm), H_mat), n)

        gk = 0.0
        for lbl in set(c0) | set(cp) | set(cm):
            sgn = np.sign(c0.get(lbl, 0.0))
            if sgn:
                gk += sgn * (cp.get(lbl, 0.0) - cm.get(lbl, 0.0)) / 2.0
        grad[k] = gk

    return grad


# ═══════════════════════════════════════════════════════════════
# §4  Expectation values  (both numpy matrix path and Qulacs path)
# ═══════════════════════════════════════════════════════════════

def _build_obs(coeffs: Dict[str, float], n: int) -> Observable:
    """Build Qulacs Observable with LSB↔MSB qubit-index fix."""
    obs = Observable(n)
    for lbl, c in coeffs.items():
        if abs(c) < 1e-12:
            continue
        parts = [f"{p} {n-1-i}" for i, p in enumerate(lbl) if p != "I"]
        obs.add_operator(float(c), " ".join(parts) if parts else "I 0")
    return obs


def _ev_numpy(psi: np.ndarray, H_mat: np.ndarray) -> float:
    return float(np.real(psi.conj() @ H_mat @ psi))


def _ev_qulacs(obs: Observable, psi: np.ndarray, n: int) -> float:
    s = QuantumState(n); s.load(psi)
    return float(np.real(obs.get_expectation_value(s)))


def compute_expectation_values(H_in:  Dict[str, float],
                                H_out: Dict[str, float]) -> dict:
    """
    Compute ⟨ψ|H|ψ⟩  and  ⟨ψ|H'|ψ⟩  for three reference states:
      |0⟩,  ground state of H,  ground state of H'

    Returns a dict with all values plus eigenvalues.
    Both numpy (matrix) and Qulacs (Observable) paths are computed.
    """
    n = validate(H_in)
    H_mat  = coeffs_to_matrix(H_in)
    Hp_mat = coeffs_to_matrix(H_out)
    H_obs  = _build_obs(H_in,  n)
    Hp_obs = _build_obs(H_out, n)

    evals_H,  evecs_H  = np.linalg.eigh(H_mat)
    evals_Hp, evecs_Hp = np.linalg.eigh(Hp_mat)

    dim = 2**n
    psi0    = np.zeros(dim, dtype=complex); psi0[0] = 1.0
    psi_gs  = evecs_H[:, 0]
    psi_gsp = evecs_Hp[:, 0]

    states = {
        "|0⟩":               psi0,
        "Ground state of H":  psi_gs,
        "Ground state of H'": psi_gsp,
    }

    rows = {}
    for name, psi in states.items():
        rows[name] = {
            "ev_H_numpy":  _ev_numpy(psi, H_mat),
            "ev_Hp_numpy": _ev_numpy(psi, Hp_mat),
            "ev_H_qulacs": _ev_qulacs(H_obs,  psi, n),
            "ev_Hp_qulacs":_ev_qulacs(Hp_obs, psi, n),
        }

    return {
        "rows":       rows,
        "evals_H":    evals_H,
        "evals_Hp":   evals_Hp,
    }


# ═══════════════════════════════════════════════════════════════
# §5  Main VQHE engine
# ═══════════════════════════════════════════════════════════════

def vqhe(
    hamiltonian: Dict[str, float],
    n_layers:    Optional[int] = None,
    n_restarts:  Optional[int] = None,
    max_iter:    int = 500,
    verbose:     bool = True,
) -> dict:
    """
    Variational Quantum Hamiltonian Engineering.

    Parameters
    ----------
    hamiltonian : dict
        Input Hamiltonian as {pauli_string: coefficient}.
        Example: {"ZZ": 1.0, "XI": 0.5, "IX": 0.5}
    n_layers : int, optional
        Ansatz layers.  Default: max(2, n_qubits).
    n_restarts : int, optional
        Random restarts.  Default: 6 for n≤2, 4 for n=3, 2 for n≥4.
    max_iter : int
        Max gradient steps per restart (default 500).
    verbose : bool
        Print progress.

    Returns
    -------
    dict with keys:
        H_original   – input Hamiltonian dict (copy)
        H_engineered – engineered Hamiltonian dict (minimised Pauli norm)
        pauli_norm_original  – ||H||_p
        pauli_norm_engineered – ||H'||_p
        norm_reduction_pct   – percentage reduction
        overhead_reduction_pct
        expectation_values   – nested dict (see compute_expectation_values)
        eigenvalues_H
        eigenvalues_Hp
    """
    n = validate(hamiltonian)
    if n_layers  is None: n_layers  = max(2, n)
    if n_restarts is None: n_restarts = 6 if n <= 2 else (4 if n == 3 else 2)

    H_mat  = coeffs_to_matrix(hamiltonian)
    ansatz = _Ansatz(n, n_layers)
    rng    = np.random.default_rng(0)

    orig_pn = pauli_norm(hamiltonian)
    D_mat   = np.diag(np.linalg.eigvalsh(H_mat))
    min_pn  = pauli_norm(matrix_to_coeffs(D_mat, n))

    if verbose:
        print(f"  n_qubits={n}  layers={n_layers}  params={ansatz.n_params}  "
              f"restarts={n_restarts}")
        print(f"  ||H||_p = {orig_pn:.6f}   "
              f"theoretical min = {min_pn:.6f}")

    best_cost, best_params = np.inf, None

    for r in range(n_restarts):
        theta0 = rng.uniform(-np.pi / 4, np.pi / 4, ansatz.n_params)

        # Stage 1: COBYLA  – gradient-free warm start
        r1 = minimize(_cost, theta0, args=(ansatz, H_mat),
                      method="COBYLA",
                      options={"maxiter": 400, "rhobeg": 0.3})

        # Stage 2: L-BFGS-B – exact gradient fine-tuning
        r2 = minimize(_cost, r1.x,   args=(ansatz, H_mat),
                      jac=_gradient,
                      method="L-BFGS-B",
                      options={"maxiter": max_iter,
                               "ftol": 1e-14, "gtol": 1e-9})

        c = r2.fun
        tag = "  ← best" if c < best_cost else ""
        if verbose:
            print(f"    restart {r+1}/{n_restarts}  "
                  f"||H'||_p = {c:.6f}{tag}")

        if c < best_cost:
            best_cost, best_params = c, r2.x.copy()

    # ── Build engineered Hamiltonian ─────────────────────────
    U_opt    = ansatz.unitary(best_params)
    H_eng    = _h_eng(U_opt, H_mat)
    eng_dict = matrix_to_coeffs(H_eng, n)
    eng_pn   = pauli_norm(eng_dict)

    red_n = (1 - eng_pn / orig_pn) * 100
    red_o = (1 - eng_pn**2 / orig_pn**2) * 100

    if verbose:
        print(f"  Final ||H'||_p = {eng_pn:.6f}  "
              f"(norm {red_n:+.2f}%,  meas. overhead {red_o:+.2f}%)")

    # ── Expectation values ───────────────────────────────────
    ev_data = compute_expectation_values(hamiltonian, eng_dict)

    return {
        "H_original":               hamiltonian,
        "H_engineered":             eng_dict,
        "pauli_norm_original":      orig_pn,
        "pauli_norm_engineered":    eng_pn,
        "pauli_norm_min_theory":    min_pn,
        "norm_reduction_pct":       red_n,
        "overhead_reduction_pct":   red_o,
        "expectation_values":       ev_data,
        "eigenvalues_H":            ev_data["evals_H"],
        "eigenvalues_Hp":           ev_data["evals_Hp"],
        "optimal_params":           best_params,
    }


# ═══════════════════════════════════════════════════════════════
# §6  Display helpers
# ═══════════════════════════════════════════════════════════════

W = 70

def _print_ham(coeffs: Dict[str, float], title: str):
    pn   = pauli_norm(coeffs)
    n    = len(next(iter(coeffs)))
    eigs = np.linalg.eigvalsh(coeffs_to_matrix(coeffs))
    if pn < 1e-12:
        return
    mx = max(abs(v) for v in coeffs.values())
    print(f"\n  ── {title} ──")
    print(f"  {'Pauli':^{n+2}}  {'Coefficient':>13}  {'|h_P|':>9}  Bar")
    print("  " + "─" * 54)
    for lbl, c in sorted(coeffs.items(), key=lambda x: -abs(x[1])):
        bar = "█" * max(1, round(abs(c) / mx * 20))
        print(f"  {lbl:^{n+2}}  {c:+13.7f}  {abs(c):9.7f}  {bar}")
    print("  " + "─" * 54)
    print(f"  {'||H||_p':^{n+2}}  {'':13}  {pn:9.7f}")
    print(f"  Eigenvalues: " + "  ".join(f"{e:+.5f}" for e in eigs))


def _print_ev(ev_data: dict):
    rows = ev_data["rows"]
    print(f"\n  ── Expectation Values ──")
    col3 = "<H'> numpy"
    col4 = "<H'> Qulacs"
    hdr = (f"  {'State':<26}"
           f"  {'<H> numpy':>11}"
           f"  {'<H> Qulacs':>12}"
           f"  {col3:>11}"
           f"  {col4:>12}")
    print(hdr)
    print("  " + "─" * 76)
    for name, v in rows.items():
        print(f"  {name:<26}"
              f"  {v['ev_H_numpy']:+11.6f}"
              f"  {v['ev_H_qulacs']:+12.6f}"
              f"  {v['ev_Hp_numpy']:+11.6f}"
              f"  {v['ev_Hp_qulacs']:+12.6f}")
    print("  " + "─" * 76)
    eH  = float(ev_data["evals_H"][0])
    eHp = float(ev_data["evals_Hp"][0])
    print(f"  Ground energy  E₀(H)  = {eH:+.7f}")
    print(f"  Ground energy  E₀(H') = {eHp:+.7f}")
    print(f"  |ΔE₀|                 = {abs(eH-eHp):.2e}  "
          + ("✓ isospectral" if abs(eH-eHp) < 1e-7 else "⚠"))


def _print_summary(res: dict):
    print(f"\n  {'═'*W}")
    print(f"  RESULT")
    print(f"  {'═'*W}")
    print(f"  Original   Pauli norm  ||H||_p   = {res['pauli_norm_original']:.7f}")
    print(f"  Engineered Pauli norm  ||H'||_p  = {res['pauli_norm_engineered']:.7f}")
    print(f"  Theoretical minimum              = {res['pauli_norm_min_theory']:.7f}")
    print(f"  Norm reduction                   = {res['norm_reduction_pct']:+.2f}%")
    print(f"  Measurement overhead reduction   = {res['overhead_reduction_pct']:+.2f}%")
    print(f"  {'═'*W}")


def print_result(res: dict):
    """Pretty-print a full VQHE result dict."""
    _print_ham(res["H_original"],   "Input Hamiltonian  H")
    _print_ham(res["H_engineered"], "Engineered Hamiltonian  H' = U†HU")
    _print_ev(res["expectation_values"])
    _print_summary(res)


# ═══════════════════════════════════════════════════════════════
# §7  Built-in Hamiltonian library
# ═══════════════════════════════════════════════════════════════

def h2_hamiltonian() -> Dict[str, float]:
    """H₂ (STO-3G, R = 0.7414 Å, Jordan-Wigner, 2 qubits)."""
    return {"II": -1.0523732, "ZI": 0.3979374, "IZ": -0.3979374,
            "ZZ": -0.0112801,  "XX":  0.1809312, "YY":  0.1809312}


def ising_hamiltonian(n: int, J: float = 1.0, h: float = 0.5) -> Dict[str, float]:
    """Transverse-field Ising: H = -J Σ ZZ − h Σ X  (open boundary)."""
    c: Dict[str, float] = {}
    for i in range(n - 1):
        k = "I"*i + "ZZ" + "I"*(n-i-2); c[k] = c.get(k, 0.0) - J
    for i in range(n):
        k = "I"*i + "X"  + "I"*(n-i-1); c[k] = c.get(k, 0.0) - h
    return c


def heisenberg_hamiltonian(n: int, J: float = 1.0) -> Dict[str, float]:
    """Heisenberg XXX: H = J Σ (XX+YY+ZZ)  (open boundary)."""
    c: Dict[str, float] = {}
    for i in range(n - 1):
        for p in "XYZ":
            k = "I"*i + p+p + "I"*(n-i-2); c[k] = c.get(k, 0.0) + J
    return c


# ═══════════════════════════════════════════════════════════════
# §8  Demo
# ═══════════════════════════════════════════════════════════════

def _demo(title: str, H: Dict[str, float]):
    print("\n" + "▓" * W)
    print(f"  {title}")
    print("▓" * W)
    res = vqhe(H, verbose=True)
    print_result(res)
    return res


if __name__ == "__main__":

    # ── Demo 1: H₂ molecule ───────────────────────────────────
    _demo("Demo 1 — H₂ Molecule  (STO-3G, 2 qubits)",
          h2_hamiltonian())

    # ── Demo 2: Transverse-field Ising (3 qubits) ────────────
    _demo("Demo 2 — Transverse-Field Ising  (3 qubits, J=1, h=0.5)",
          ising_hamiltonian(n=3, J=1.0, h=0.5))

    # ── Demo 3: Heisenberg chain (3 qubits) ───────────────────
    _demo("Demo 3 — Heisenberg XXX Chain  (3 qubits)",
          heisenberg_hamiltonian(n=3, J=1.0))

    # ── Demo 4: Custom user Hamiltonian ───────────────────────
    _demo("Demo 4 — Custom Hamiltonian  (supply your own dict)",
          {"ZZ": 1.0, "XI": 0.4, "IX": 0.4,
           "YY": -0.3, "ZI": 0.2, "IZ": -0.2})
