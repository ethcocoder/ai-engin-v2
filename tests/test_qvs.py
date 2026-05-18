import sys
import os
import numpy as np
import unittest

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qau_qvs.core.qvs import QVS
from qau_qvs.core.asc import ASC
from qau_qvs.core.ncb import NCB
from qau_qvs.fields.quantum_fields import (
    QuantumAlgorithms, QuantumErrorCorrection, QuantumSimulation,
    QuantumCryptography, QuantumFieldTheory, QuantumMachineLearning
)

class TestQVS(unittest.TestCase):
    def setUp(self):
        self.qvs = QVS()

    def test_jit_unitary_fusion(self):
        """Verify that sequential ROTATE instructions are fused and applied correctly."""
        psi_id = self.qvs.create_asc(size=1)
        # Apply two Pauli-X gates (identity)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        self.qvs.ROTATE(psi_id, X)
        self.qvs.ROTATE(psi_id, X)
        
        # JIT cache should have 2 matrices
        self.assertEqual(len(self.qvs.pending_rotations[psi_id]), 2)
        
        # Calling get_asc should trigger the fusion and execution
        asc = self.qvs.get_asc(psi_id)
        self.assertEqual(len(self.qvs.pending_rotations[psi_id]), 0)
        
        # |0> X X = |0>
        self.assertAlmostEqual(abs(asc.amplitudes[(0,)])**2, 1.0)

    def test_quantum_trajectories(self):
        """Verify the Monte Carlo trajectory simulation."""
        psi_id = self.qvs.create_asc(size=1)
        # Create equal superposition
        self.qvs.SUPERPOSE(psi_id, [(0,), (1,)])
        
        # Run 500 trajectories
        counts = self.qvs.run_trajectories(psi_id, trials=500)
        
        # Should be roughly 0.5 each
        self.assertAlmostEqual(counts[(0,)], 0.5, delta=0.1)
        self.assertAlmostEqual(counts[(1,)], 0.5, delta=0.1)

    # ... [Keep existing tests for coverage] ...

    def test_superposition(self):
        psi_id = self.qvs.create_asc(size=1)
        self.qvs.SUPERPOSE(psi_id, [(0,), (1,)])
        asc = self.qvs.get_asc(psi_id)
        self.assertAlmostEqual(abs(asc.amplitudes[(0,)])**2, 0.5)

    def test_interference(self):
        psi_id = self.qvs.create_asc(size=1)
        self.qvs.SUPERPOSE(psi_id, [(1,)])
        self.qvs.WEAVE(psi_id, (0,), np.pi/2)
        asc = self.qvs.get_asc(psi_id)
        self.assertAlmostEqual(asc.amplitudes[(1,)].imag, 1, places=10)

    def test_entanglement(self):
        alice_id = self.qvs.create_asc(size=1)
        bob_id = self.qvs.create_asc(size=1)
        bonded_id = self.qvs.BOND(alice_id, bob_id, "bell")
        asc = self.qvs.get_asc(bonded_id)
        entropy = NCB.get_entanglement_entropy(asc, partition_idx=1)
        self.assertAlmostEqual(entropy, 1.0) 

    def test_quantum_fields(self):
        alg = QuantumAlgorithms(self.qvs)
        res_alg = alg.shor_factorization_pattern()
        self.assertIsInstance(res_alg, tuple)
        
        crypto = QuantumCryptography(self.qvs)
        alice_key, bob_key = crypto.e91_key_exchange()
        self.assertEqual(alice_key, bob_key)

if __name__ == '__main__':
    unittest.main()
