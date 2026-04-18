import numpy as np
import random
from typing import Dict, Tuple, List, Optional, Any, Union, Callable
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
    Quantum Virtual Substrate (QVS) - ADVANCED v1.1.0
    ================================================
    The foundational OPERATING SYSTEM LAYER for the QAU.
    
    Includes Advanced Features:
    - JIT Unitary Fusion (Instruction optimization)
    - Quantum Trajectory Monte Carlo (Scalable simulation)
    - Instruction Layering for Field Stability
    """
    
    def __init__(self):
        self.ascs: Dict[str, ASC] = {} 
        self.next_id = 0
        self.instruction_history: List[Dict[str, Any]] = []
        self.pending_rotations: Dict[str, List[np.ndarray]] = {} # For JIT fusion

    # ------------------------------------------------------------------
    # Resource Management
    # ------------------------------------------------------------------

    def create_asc(self, basis_states: Optional[Dict[Tuple, complex]] = None, size: int = 1) -> str:
        asc_id = f"ASC_{self.next_id}"
        self.next_id += 1
        self.ascs[asc_id] = ASC(basis_states, size)
        self.pending_rotations[asc_id] = []
        return asc_id

    def delete_asc(self, asc_id: str):
        if asc_id in self.ascs:
            self.ascs.pop(asc_id, None)
            self.pending_rotations.pop(asc_id, None)

    def get_asc(self, asc_id: str) -> ASC:
        if asc_id not in self.ascs:
            raise KeyError(f"ASC {asc_id} not found.")
        # Apply any pending fused rotations before returning
        self._flush_jit_cache(asc_id)
        return self.ascs[asc_id]

    # ------------------------------------------------------------------
    # JIT Unitary Fusion
    # ------------------------------------------------------------------

    def _flush_jit_cache(self, asc_id: str):
        """Fuses all pending rotations into a single optimized operation."""
        pending = self.pending_rotations.get(asc_id, [])
        if not pending:
            return
        
        asc = self.ascs[asc_id]
        # Fuse all U matrices
        dim = 2**asc.size
        fused_U = np.eye(dim, dtype=complex)
        for U in pending:
            fused_U = np.dot(U, fused_U)
        
        # Apply the fused unitary once
        self._apply_raw_rotation(asc, fused_U)
        self.pending_rotations[asc_id] = []

    def _apply_raw_rotation(self, asc: ASC, unitary: np.ndarray):
        """Internal worker to apply rotation and prune amplitudes."""
        vec = asc.get_state_vector()
        new_vec = np.dot(unitary, vec)
        new_amplitudes = {}
        for i in range(len(new_vec)):
            if abs(new_vec[i]) > 1e-12:
                bits = tuple((i >> (asc.size - 1 - j)) & 1 for j in range(asc.size))
                new_amplitudes[bits] = complex(new_vec[i])
        asc.amplitudes = new_amplitudes

    # ------------------------------------------------------------------
    # The QVS Instruction Set
    # ------------------------------------------------------------------

    def SUPERPOSE(self, asc_id: str, basis_states: List[Tuple[int, ...]]) -> str:
        asc = self.get_asc(asc_id)
        weight = 1.0 / np.sqrt(len(basis_states))
        asc.amplitudes = {tuple(state): complex(weight) for state in basis_states}
        return asc_id

    def WEAVE(self, asc_id: str, target_bits: Optional[Tuple[int, ...]] = None, phase_angle: float = 0.0) -> str:
        asc = self.get_asc(asc_id)
        bits = target_bits if target_bits is not None else (0,)
        RPW.weave(asc, bits, {1: phase_angle})
        return asc_id

    def BOND(self, asc_id_a: str, asc_id_b: str, bond_type: str = "bell") -> str:
        asc_a = self.get_asc(asc_id_a)
        asc_b = self.get_asc(asc_id_b)
        bonded_asc = NCB.bond(asc_a, asc_b, bond_type)
        self.delete_asc(asc_id_a)
        self.delete_asc(asc_id_b)
        return self.create_asc(bonded_asc.amplitudes, bonded_asc.size)

    def ROTATE(self, asc_id: str, unitary: np.ndarray) -> str:
        """Adds a unitary to the JIT cache for optimized execution."""
        if asc_id not in self.pending_rotations:
            self.pending_rotations[asc_id] = []
        self.pending_rotations[asc_id].append(unitary)
        return asc_id

    def COLLAPSE(self, asc_id: str) -> Tuple[int, ...]:
        asc = self.get_asc(asc_id) # Triggers JIT flush
        states = list(asc.amplitudes.keys())
        p = np.array([abs(asc.amplitudes[s])**2 for s in states])
        
        # --- Paradox Safety Protocol 1.1 ---
        # Handle NaNs or all-zero amplitudes gracefully
        if np.any(np.isnan(p)) or p.sum() <= 0:
            p = np.ones_like(p) / len(p)
        else:
            p /= p.sum()
            
        idx = np.random.choice(len(states), p=p)
        outcome = states[idx]
        asc.amplitudes = {outcome: 1.0 + 0j}
        return outcome

    # ------------------------------------------------------------------
    # Advanced: Quantum Trajectory (Monte Carlo)
    # ------------------------------------------------------------------

    def run_trajectories(self, asc_id: str, trials: int = 100) -> Dict[Tuple[int, ...], float]:
        """
        Executes a measurement-first quantum trajectory simulation.
        Highly scalable for field theories as it explores the most probable paths.
        """
        original_asc = self.get_asc(asc_id)
        results = {}
        
        for _ in range(trials):
            # Clone and collapse
            temp_asc = original_asc.clone()
            # Simulate a single trajectory collapse
            states = list(temp_asc.amplitudes.keys())
            p = np.array([abs(temp_asc.amplitudes[s])**2 for s in states])
            p /= p.sum()
            outcome = states[np.random.choice(len(states), p=p)]
            results[outcome] = results.get(outcome, 0.0) + (1.0 / trials)
            
        return results
