import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
import copy

class ASC:
    """
    Amplitude Superposition Cell (ASC) - TPU Optimized v2.0
    =======================================================
    The primitive of COHERENT MULTIPLICITY.
    
    Now uses Torch Complex Tensors for massive speedup on TPU/GPU hardware.
    """

    def __init__(self, amplitudes: Optional[Dict[Tuple, complex]] = None, size: Optional[int] = None, device: str = "cpu"):
        self.size: int = size if size is not None else 0
        self.device = device
        dim = 2 ** self.size
        
        # Dense State Vector for TPU speed (Complex64)
        self.vec = torch.zeros(dim, dtype=torch.complex64, device=device)
        
        if amplitudes:
            for state, weight in amplitudes.items():
                idx = sum(bit * (2 ** (self.size - 1 - i)) for i, bit in enumerate(state))
                self.vec[idx] = weight
        elif self.size > 0:
            # Default to |0...0> ground state
            self.vec[0] = 1.0 + 0j

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def normalize(self) -> "ASC":
        """Renormalize so sum(|alpha|^2) == 1."""
        norm = torch.norm(self.vec)
        if norm > 1e-15:
            self.vec = self.vec / norm
        return self

    def prune(self, threshold: float = 1e-12) -> "ASC":
        """Remove amplitudes below threshold by zeroing them out."""
        mask = torch.abs(self.vec) > threshold
        self.vec = self.vec * mask.to(self.vec.dtype)
        return self

    def get_state_vector(self) -> torch.Tensor:
        return self.vec

    def get_density_matrix(self) -> torch.Tensor:
        """Return the density matrix rho = |psi><psi|."""
        return torch.outer(self.vec, self.vec.conj())

    def fidelity(self, other: "ASC") -> float:
        """Compute |<self|other>|^2 — how similar two states are."""
        inner_prod = torch.dot(self.vec.conj(), other.vec)
        return float(torch.abs(inner_prod) ** 2)

    def expectation_value(self, observable: torch.Tensor) -> float:
        """Compute <psi|O|psi> for a given Hermitian observable matrix."""
        # Ensure observable is on the same device
        obs = observable.to(self.device).to(torch.complex64)
        val = torch.dot(self.vec.conj(), torch.mv(obs, self.vec))
        return float(val.real)

    def entropy(self) -> float:
        """Von Neumann entropy for pure state (calculated on probs)."""
        probs = torch.abs(self.vec) ** 2
        probs = probs[probs > 1e-15]
        return float(-torch.sum(probs * torch.log2(probs)))

    def clone(self) -> "ASC":
        new_asc = ASC(size=self.size, device=self.device)
        new_asc.vec = self.vec.clone()
        return new_asc

    def to(self, device: str) -> "ASC":
        """Moves the underlying tensor to the target device (TPU/GPU)."""
        self.device = device
        self.vec = self.vec.to(device)
        return self

    @property
    def amplitudes(self) -> Dict[Tuple, complex]:
        """Legacy compatibility: returns non-zero amplitudes as dict."""
        amps = {}
        vec_np = self.vec.cpu().numpy()
        for i in range(len(vec_np)):
            if abs(vec_np[i]) > 1e-12:
                bits = tuple((i >> (self.size - 1 - j)) & 1 for j in range(self.size))
                amps[bits] = complex(vec_np[i])
        return amps

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"ASC(size={self.size}, device={self.device}, active={torch.sum(torch.abs(self.vec) > 1e-12).item()})"

    def __len__(self) -> int:
        return 2 ** self.size
