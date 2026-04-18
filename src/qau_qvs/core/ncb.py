import numpy as np
import copy
from typing import Tuple, Dict, List, Optional
from .asc import ASC

class NCB:
    """
    Non-Local Correlation Bond (NCB)
    ================================
    The primitive of INFORMATIONAL CONSTRAINT.
    
    A bond between two or more ASCs that ensures joint probability distributions 
    cannot be factorized. On silicon, this is represented by a shared pointer: 
    when one ASC changes, its bonded partner(s) change synchronously.
    
    This is effectively the mechanism behind quantum entanglement.
    """
    
    def __init__(self, asc_ids: List[str], correlation_type: str = "custom"):
        """
        Args:
            asc_ids: List of ASC IDs participating in the bond.
            correlation_type: 'bell', 'ghz', 'w-state', or 'custom'.
        """
        self.asc_ids = asc_ids
        self.correlation_type = correlation_type

    @staticmethod
    def bond(asc_a: ASC, asc_b: ASC, bond_type: str = "bell") -> ASC:
        """
        Forges a bond between two systems and returns the unified joint state.
        
        This mimics the QVS behavior where the systems 'couple' and become 
        one computational object.
        """
        # 1. Start with tensor product of A and B
        joint_size = asc_a.size + asc_b.size
        joint_amplitudes: Dict[Tuple, complex] = {}
        for (sa, wa) in asc_a.amplitudes.items():
            for (sb, wb) in asc_b.amplitudes.items():
                joint_state = sa + sb
                joint_amplitudes[joint_state] = wa * wb
        
        # 2. Forge the bond (Apply ENTAGLEMENT)
        if bond_type == "bell":
            # Typical Bell state: |00> + |11> or similar.
            # In silicon-native terms, this is a SHARED constraint.
            # Here we simulate it by transforming the joint state.
            if joint_size >= 2:
                # Let's create |00...0> + |11...0>
                new_amps = {}
                base_weight = 1.0 / np.sqrt(2)
                
                # Default to base states |0...0> and |1...1...0...>
                s0 = (0,) * joint_size
                s1 = (1, 1) + (0,) * (joint_size - 2)
                new_amps[s0] = base_weight
                new_amps[s1] = base_weight
                joint_amplitudes = new_amps

        elif bond_type == "ghz":
            # Global non-local constraint: |000...> + |111...>
            if joint_size >= 2:
                base_weight = 1.0 / np.sqrt(2)
                s0 = (0,) * joint_size
                s1 = (1,) * joint_size
                joint_amplitudes = {s0: base_weight, s1: base_weight}

        return ASC(joint_amplitudes, joint_size).normalize()

    @staticmethod
    def get_entanglement_entropy(asc: ASC, partition_idx: int) -> float:
        """
        Calculate the entanglement entropy of an ASC's partition.
        Measure of how strong the NCB is across a split in the system.
        """
        # For simplicity, we convert to full density matrix and then partial trace.
        # Silicon-native would do this via tensor contraction.
        vec = asc.get_state_vector()
        dim_a = 2**partition_idx
        dim_b = 2**(asc.size - partition_idx)
        
        mat = vec.reshape(dim_a, dim_b)
        U, S, Vh = np.linalg.svd(mat) # SVD decomposition for Schmidt coefficients
        
        # Reduced density matrix eigenvalues are S^2
        probs = S**2
        probs = probs[probs > 1e-15]
        return float(-np.sum(probs * np.log2(probs)))
