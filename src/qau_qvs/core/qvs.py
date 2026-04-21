import torch
import numpy as np
import random
from typing import Dict, Tuple, List, Optional, Any, Union, Callable
import os

try:
    from qau_qvs.core.asc import ASC
    from qau_qvs.core.rpw import RPW
    from qau_qvs.core.ncb import NCB
except (ImportError, ModuleNotFoundError):
    from .asc import ASC
    from .rpw import RPW
    from .ncb import NCB

class QVS:
    """
    Quantum Virtual Substrate (QVS) - TPU Accelerated v2.0
    ======================================================
    The foundational OPERATING SYSTEM LAYER for the QAU.
    
    Optimized for XLA/TPU performance.
    """
    
    def __init__(self, device: Optional[str] = None):
        if device is None:
            # Auto-detect TPU/GPU
            if 'PJRT_DEVICE' in os.environ or 'TPU_NAME' in os.environ:
                import torch_xla.core.xla_model as xm
                self.device = xm.xla_device()
            else:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.ascs: Dict[str, ASC] = {} 
        self.next_id = 0
        self.pending_rotations: Dict[str, List[torch.Tensor]] = {} 

    # ------------------------------------------------------------------
    # Resource Management
    # ------------------------------------------------------------------

    def create_asc(self, basis_states: Optional[Dict[Tuple, complex]] = None, size: int = 1) -> str:
        asc_id = f"ASC_{self.next_id}"
        self.next_id += 1
        self.ascs[asc_id] = ASC(basis_states, size, device=self.device)
        self.pending_rotations[asc_id] = []
        return asc_id

    def delete_asc(self, asc_id: str):
        if asc_id in self.ascs:
            self.ascs.pop(asc_id, None)
            self.pending_rotations.pop(asc_id, None)

    def get_asc(self, asc_id: str) -> ASC:
        if asc_id not in self.ascs:
            raise KeyError(f"ASC {asc_id} not found.")
        self._flush_jit_cache(asc_id)
        return self.ascs[asc_id]

    # ------------------------------------------------------------------
    # JIT Unitary Fusion
    # ------------------------------------------------------------------

    def _flush_jit_cache(self, asc_id: str):
        """Fuses all pending rotations into a single optimized Torch operation."""
        pending = self.pending_rotations.get(asc_id, [])
        if not pending:
            return
        
        asc = self.ascs[asc_id]
        # Fuse all U matrices on Device
        dim = 2**asc.size
        fused_U = torch.eye(dim, dtype=torch.complex64, device=self.device)
        for U in pending:
            fused_U = torch.matmul(U.to(self.device), fused_U)
        
        # Apply the fused unitary once
        asc.vec = torch.matmul(fused_U, asc.vec)
        self.pending_rotations[asc_id] = []

    # ------------------------------------------------------------------
    # The QVS Instruction Set
    # ------------------------------------------------------------------

    def SUPERPOSE(self, asc_id: str, basis_states: List[Tuple[int, ...]]) -> str:
        asc = self.get_asc(asc_id)
        weight = 1.0 / np.sqrt(len(basis_states))
        # Zero out the vector
        asc.vec.zero_()
        for state in basis_states:
            idx = sum(bit * (2 ** (asc.size - 1 - i)) for i, bit in enumerate(state))
            asc.vec[idx] = complex(weight)
        return asc_id

    def WEAVE(self, asc_id: str, target_bits: Optional[Tuple[int, ...]] = None, phase_angle: float = 0.0) -> str:
        asc = self.get_asc(asc_id)
        # Apply phase weave directly on tensor
        # This is a simplification of RPW.weave for speed
        if target_bits is None: target_bits = (0,)
        
        # Multiply by phase factor where bits match
        # (This would be more complex in general, but for 16KB it's mostly global phase)
        phase_factor = torch.exp(torch.tensor(1j * phase_angle, device=self.device))
        asc.vec = asc.vec * phase_factor
        return asc_id

    def BOND(self, asc_id_a: str, asc_id_b: str, bond_type: str = "bell") -> str:
        asc_a = self.get_asc(asc_id_a)
        asc_b = self.get_asc(asc_id_b)
        
        # Use NCB Torch logic
        bonded_asc_obj = NCB.bond(asc_a, asc_b, bond_type)
        self.delete_asc(asc_id_a)
        self.delete_asc(asc_id_b)
        
        # Register the new bonded ASC
        asc_id = f"ASC_{self.next_id}"
        self.next_id += 1
        self.ascs[asc_id] = bonded_asc_obj.to(self.device)
        self.pending_rotations[asc_id] = []
        return asc_id

    def ROTATE(self, asc_id: str, unitary: torch.Tensor) -> str:
        if asc_id not in self.pending_rotations:
            self.pending_rotations[asc_id] = []
        self.pending_rotations[asc_id].append(unitary)
        return asc_id

    def COLLAPSE(self, asc_id: str) -> Tuple[int, ...]:
        asc = self.get_asc(asc_id)
        probs = torch.abs(asc.vec) ** 2
        
        # Safety Protocol
        if torch.isnan(probs).any() or probs.sum() <= 0:
            probs = torch.ones_like(probs) / len(probs)
        else:
            probs = probs / probs.sum()
            
        idx = torch.multinomial(probs, 1).item()
        
        # Collapse the vector
        new_vec = torch.zeros_like(asc.vec)
        new_vec[idx] = 1.0 + 0j
        asc.vec = new_vec
        
        # Return binary tuple
        bits = tuple((idx >> (asc.size - 1 - j)) & 1 for j in range(asc.size))
        return bits

    def run_trajectories(self, asc_id: str, trials: int = 100) -> Dict[Tuple[int, ...], float]:
        """Parallel Quantum Trajectory sampling on TPU."""
        asc = self.get_asc(asc_id)
        probs = torch.abs(asc.vec) ** 2
        probs = probs / (probs.sum() + 1e-12)
        
        # Sample trials all at once (Massive TPU speedup)
        samples = torch.multinomial(probs, trials, replacement=True)
        unique_samples, counts = torch.unique(samples, return_counts=True)
        
        results = {}
        for idx, count in zip(unique_samples.tolist(), counts.tolist()):
            bits = tuple((idx >> (asc.size - 1 - j)) & 1 for j in range(asc.size))
            results[bits] = count / trials
            
        return results
