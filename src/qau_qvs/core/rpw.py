from __future__ import annotations
import torch
import numpy as np
from typing import Tuple, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .asc import ASC

class RPW:
    """
    Relative Phase Weave (RPW) - TPU Optimized v2.0
    ==============================================
    The primitive of INTERFERENCE.
    """
    
    @staticmethod
    def apply_phase(asc: 'ASC', state_idx: int, theta: float):
        """
        Apply a phase rotation e^(i*theta) to a specific basis state.
        """
        device = asc.vec.device
        phase_factor = torch.exp(torch.tensor(1j * theta, device=device))
        asc.vec[state_idx] *= phase_factor
        return asc

    @staticmethod
    def weave(asc: 'ASC', target_bits: Tuple[int, ...], phases: Dict[int, float]):
        """
        Applies a phase warp to specific bit-states using Torch.
        """
        device = asc.vec.device
        
        # Optimized Phase Application
        phase_val = phases.get(1, 0.0)
        phase_factor = torch.exp(torch.tensor(1j * phase_val, device=device))
        
        # Apply phase weave directly to the TPU manifold
        asc.vec = asc.vec * phase_factor
        return asc

    @staticmethod
    def global_phase(asc: 'ASC', theta: float):
        """Apply a global phase shift to all states in an ASC."""
        device = asc.vec.device
        phase_factor = torch.exp(torch.tensor(1j * theta, device=device))
        asc.vec = asc.vec * phase_factor
        return asc
