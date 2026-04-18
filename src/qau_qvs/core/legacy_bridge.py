import numpy as np
from .qvs import QVS

class LegacyBridge:
    """
    Legacy Bridge (QISK-B)
    ======================
    Adapts traditional gate-based quantum circuits to the 
    high-performance Quantum Virtual Substrate (QVS).
    
    This allows researchers to bring their 'legacy' algorithms 
    (designed for NISQ machines) and execute them natively on the QAU.
    """
    
    def __init__(self, qvs: QVS):
        self.qvs = qvs

    def hadamard(self, asc_id: str, bit_idx: int) -> str:
        """Legacy Hadamard gate: H = 1/√2 [[1, 1], [1, -1]]"""
        H = (1.0 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
        # Construct full unitary for rotation
        asc = self.qvs.get_asc(asc_id)
        U = self._expand_gate(H, bit_idx, asc.size)
        return self.qvs.ROTATE(asc_id, U)

    def cnot(self, asc_id: str, control: int, target: int) -> str:
        """Legacy CNOT gate maps to a ROTATE operation on the QVS."""
        asc = self.qvs.get_asc(asc_id)
        dim = 2**asc.size
        U = np.eye(dim, dtype=complex)
        for i in range(dim):
            # Check if control bit is 1
            if (i >> (asc.size - 1 - control)) & 1:
                # Flip target bit
                target_bit_mask = 1 << (asc.size - 1 - target)
                j = i ^ target_bit_mask
                U[i, i] = 0
                U[i, j] = 1
        return self.qvs.ROTATE(asc_id, U)

    def _expand_gate(self, gate: np.ndarray, bit_idx: int, total_size: int) -> np.ndarray:
        """Helper to lift a single-bit gate to the full Hilbert space."""
        I = np.eye(2)
        operators = [I] * total_size
        operators[bit_idx] = gate
        
        full_U = operators[0]
        for op in operators[1:]:
            full_U = np.kron(full_U, op)
        return full_U
