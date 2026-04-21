import torch
import numpy as np
from typing import Tuple, Dict, List, Optional
from .asc import ASC

class NCB:
    """
    Non-Local Correlation Bond (NCB) - TPU Optimized
    ================================================
    Handles informational entanglement between latent channels.
    """
    
    @staticmethod
    def bond(asc_a: ASC, asc_b: ASC, bond_type: str = "bell") -> ASC:
        """
        Forges a bond using Torch Kronecker products.
        """
        device = asc_a.device
        joint_size = asc_a.size + asc_b.size
        
        # 1. Base joint state (Tensor Product)
        # torch.kron is available in recent torch versions or we can use outer + reshape
        vec_a = asc_a.vec
        vec_b = asc_b.vec
        joint_vec = torch.outer(vec_a, vec_b).reshape(-1)
        
        # 2. Apply ENTAGLEMENT Logic directly on tensor
        if bond_type == "bell":
            if joint_size >= 2:
                # Force into |00> + |11> state representation
                new_vec = torch.zeros(2**joint_size, dtype=torch.complex64, device=device)
                new_vec[0] = 0.7071 # |00...0>
                # index for |110...0>
                idx_11 = 3 * (2**(joint_size - 2))
                new_vec[idx_11] = 0.7071
                joint_vec = new_vec

        elif bond_type == "ghz":
            if joint_size >= 2:
                new_vec = torch.zeros(2**joint_size, dtype=torch.complex64, device=device)
                new_vec[0] = 0.7071
                new_vec[-1] = 0.7071 # |11...1>
                joint_vec = new_vec

        res_asc = ASC(size=joint_size, device=device)
        res_asc.vec = joint_vec
        return res_asc.normalize()

    @staticmethod
    def get_entanglement_entropy(asc: ASC, partition_idx: int) -> float:
        """
        Calculate entropy using Torch SVD (highly optimized on TPU).
        """
        vec = asc.vec
        dim_a = 2**partition_idx
        dim_b = 2**(asc.size - partition_idx)
        
        mat = vec.reshape(dim_a, dim_b)
        # SVD on TPU
        _, S, _ = torch.linalg.svd(mat)
        
        probs = torch.abs(S)**2
        probs = probs[probs > 1e-15]
        return float(-torch.sum(probs * torch.log2(probs)))
