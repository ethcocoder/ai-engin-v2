import numpy as np
from typing import Tuple, Dict

class RPW:
    """
    Relative Phase Weave (RPW)
    =========================
    The primitive of INTERFERENCE.
    
    In the QAU architecture, the RPW handles the phase relationships 
    between basis states (e^iθ). Using geometric rotor algebra 
    (represented here by complex rotations) to enable constructive 
    and destructive amplification of quantum paths.
    """
    
    def __init__(self, angle: float = 0.0):
        self.angle = angle # theta

    @staticmethod
    def apply_phase(asc, state: Tuple, theta: float):
        """
        Apply a phase rotation e^(i*theta) to a specific basis state.
        In silicon, this is a single FMA operation on the complex coefficients.
        """
        if state in asc.amplitudes:
            # Rotor R = cos(theta) + i*sin(theta)
            asc.amplitudes[state] *= np.exp(1j * theta)
        return asc

    @staticmethod
    def weave(asc, target_bits: Tuple[int, ...], phase_map: Dict[int, float]):
        """
        Create a complex interference pattern (WEAVE) across multiple bits.
        This is a high-level instruction that maps to multiple RPW rotations.
        
        Args:
            asc: The ASC to weave.
            target_bits: Indices of qubits to apply phase to.
            phase_map: Mapping from bit-value (0/1) to phase shift.
        """
        for state in list(asc.amplitudes.keys()):
            total_phase = 0.0
            for bit_idx in target_bits:
                bit_val = state[bit_idx]
                total_phase += phase_map.get(bit_val, 0.0)
            
            if total_phase != 0:
                asc.amplitudes[state] *= np.exp(1j * total_phase)
        return asc

    @staticmethod
    def global_phase(asc, theta: float):
        """Apply a global phase shift to all states in an ASC."""
        phase_factor = np.exp(1j * theta)
        for state in asc.amplitudes:
            asc.amplitudes[state] *= phase_factor
        return asc
