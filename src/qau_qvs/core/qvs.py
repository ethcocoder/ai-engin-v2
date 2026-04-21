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
    Optimized for XLA/TPU performance. No Python-on-Tensor checks.
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

    def _flush_jit_cache(self, asc_id: str):
        pending = self.pending_rotations.get(asc_id, [])
        if not pending: return
        asc = self.ascs[asc_id]
        dim = 2**asc.size
        fused_U = torch.eye(dim, dtype=torch.complex64, device=self.device)
        for U in pending:
            fused_U = torch.matmul(U.to(self.device), fused_U)
        asc.vec = torch.matmul(fused_U, asc.vec)
        self.pending_rotations[asc_id] = []

    def SUPERPOSE(self, asc_id: str, basis_states: List[Tuple[int, ...]]) -> str:
        asc = self.get_asc(asc_id)
        weight = 1.0 / np.sqrt(len(basis_states))
        asc.vec.zero_()
        for state in basis_states:
            idx = sum(bit * (2 ** (asc.size - 1 - i)) for i, bit in enumerate(state))
            asc.vec[idx] = complex(weight)
        return asc_id

    def WEAVE(self, asc_id: str, target_bits: Optional[Tuple[int, ...]] = None, phase_angle: float = 0.0) -> str:
        asc = self.get_asc(asc_id)
        phase_factor = torch.exp(torch.tensor(1j * phase_angle, device=self.device, dtype=torch.complex64))
        asc.vec = asc.vec * phase_factor
        return asc_id

    def BOND(self, asc_id_a: str, asc_id_b: str, bond_type: str = "bell") -> str:
        asc_a = self.get_asc(asc_id_a)
        asc_b = self.get_asc(asc_id_b)
        bonded_asc_obj = NCB.bond(asc_a, asc_b, bond_type)
        self.delete_asc(asc_id_a)
        self.delete_asc(asc_id_b)
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

    def batch_COLLAPSE(self, asc_ids: List[str]) -> torch.Tensor:
        """Vectorized collapse for an entire batch. Zero Sync Points."""
        amps = torch.stack([self.ascs[aid].vec for aid in asc_ids])
        # Use the device of the amplitudes
        device = amps.device
        probs = torch.abs(amps).pow(2).to(torch.float32)
        
        # TPU Safety
        has_nan = torch.isnan(probs).any()
        sum_p = probs.sum(dim=-1, keepdim=True)
        mask = (has_nan | (sum_p <= 1e-8))
        probs = torch.where(mask, torch.ones_like(probs) / (probs.shape[-1] + 1e-8), probs / (sum_p + 1e-8))
        
        # Batch Multinomial
        indices = torch.multinomial(probs, 1).squeeze(-1)
        
        # Apply parity logic (Universal for Bell states)
        outcomes = (indices % 2 == 0).float() * 2.0 - 1.0
        return outcomes

    def batch_run_trajectories(self, intensities: torch.Tensor, trials: int = 10) -> torch.Tensor:
        """
        Hyper-Speed Quantum Trajectory Sampling.
        Dynamic Device Detection (TPU/GPU/CPU).
        """
        device = intensities.device
        # Generate stochastic bias aligned with neural intensities
        B = intensities.shape[0]
        # Simulate p = cos^2(theta) behavior of quantum interference
        random_seed = torch.rand(B, trials, device=device)
        threshold = torch.cos(intensities.unsqueeze(-1) * np.pi).pow(2)
        
        # Parallel Realities
        realities = torch.where(random_seed < threshold, 1.0, -1.0)
        return realities.mean(dim=-1)
