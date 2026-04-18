import numpy as np
from typing import Dict, Tuple, Optional, List
import copy


class ASC:
    """
    Amplitude Superposition Cell (ASC)
    ===================================
    The primitive of COHERENT MULTIPLICITY.

    A lazy tensor block that stores only non-zero amplitudes (sparse representation),
    yet performs arithmetic as if it were a full 2^n complex vector.

    Key properties:
    - Memory-efficient: only non-zero amplitudes are stored.
    - Mathematically complete: represents a full quantum state vector.
    - Normalizable: sum(|alpha_i|^2) == 1 at all times.
    """

    def __init__(self, amplitudes: Optional[Dict[Tuple, complex]] = None, size: Optional[int] = None):
        """
        Args:
            amplitudes: dict mapping basis state tuple -> complex amplitude.
            size: number of qubits this ASC represents.
        """
        self.amplitudes: Dict[Tuple, complex] = amplitudes if amplitudes is not None else {}
        self.size: int = size if size is not None else 0

        # Default to |0...0> ground state if nothing provided
        if not self.amplitudes and self.size > 0:
            self.amplitudes[(0,) * self.size] = 1.0 + 0j

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def normalize(self) -> "ASC":
        """Renormalize so sum(|alpha|^2) == 1."""
        norm = np.sqrt(sum(abs(a) ** 2 for a in self.amplitudes.values()))
        if norm > 1e-15:
            self.amplitudes = {k: v / norm for k, v in self.amplitudes.items()}
        return self

    def prune(self, threshold: float = 1e-12) -> "ASC":
        """Remove amplitudes below threshold (decoherence / gate noise)."""
        self.amplitudes = {k: v for k, v in self.amplitudes.items() if abs(v) > threshold}
        return self

    def get_state_vector(self) -> np.ndarray:
        """Return the full 2^n state vector (for small systems / debugging)."""
        dim = 2 ** self.size
        vec = np.zeros(dim, dtype=complex)
        for state, weight in self.amplitudes.items():
            idx = sum(bit * (2 ** (self.size - 1 - i)) for i, bit in enumerate(state))
            vec[idx] = weight
        return vec

    def get_density_matrix(self) -> np.ndarray:
        """Return the density matrix rho = |psi><psi|."""
        vec = self.get_state_vector()
        return np.outer(vec, vec.conj())

    def fidelity(self, other: "ASC") -> float:
        """Compute |<self|other>|^2 — how similar two states are."""
        v1 = self.get_state_vector()
        v2 = other.get_state_vector()
        return float(abs(np.dot(v1.conj(), v2)) ** 2)

    def expectation_value(self, observable: np.ndarray) -> float:
        """Compute <psi|O|psi> for a given Hermitian observable matrix."""
        vec = self.get_state_vector()
        return float(np.real(vec.conj() @ observable @ vec))

    def entropy(self) -> float:
        """Von Neumann entropy S = -sum(p log p) for this pure state (always 0 for pure)."""
        probs = np.array([abs(a) ** 2 for a in self.amplitudes.values()])
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))

    def clone(self) -> "ASC":
        """Deep-copy this ASC (no quantum no-cloning violation — this is classical sim)."""
        return ASC(copy.deepcopy(self.amplitudes), self.size)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        top = sorted(self.amplitudes.items(), key=lambda x: -abs(x[1]) ** 2)[:3]
        s = ", ".join(f"|{''.join(map(str,k))}⟩: {v:.3f}" for k, v in top)
        return f"ASC(size={self.size}, states={len(self.amplitudes)}, top=[{s}])"

    def __len__(self) -> int:
        return len(self.amplitudes)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ASC):
            return NotImplemented
        if self.size != other.size:
            return False
        for k in set(self.amplitudes) | set(other.amplitudes):
            if abs(self.amplitudes.get(k, 0) - other.amplitudes.get(k, 0)) > 1e-9:
                return False
        return True
