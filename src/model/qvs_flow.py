import torch
import torch.nn as nn

class QVSUnitaryCoupling(nn.Module):
    """
    Quantum Virtual Substrate (QVS) Unitary Coupling Layer.
    Uses Cayley parametrization to produce a strictly orthogonal (unitary) matrix.
    Optimized with caching and numerical stability guards.
    """
    def __init__(self, channels, max_A_norm=2.0):
        super().__init__()
        self.channels = channels
        self.max_A_norm = max_A_norm
        
        # Skew-symmetric matrix parameters: A = A_params - A_params^T
        self.A_params = nn.Parameter(torch.randn(channels, channels) * 0.01)
        
        # FIX 1: Cached orthogonal matrix to prevent redundant O(C^3) inversions
        self.register_buffer('_W_cache', torch.eye(channels))
        self._cache_valid = False
    
    def _compute_orthogonal_matrix(self):
        """
        Compute W = (I - A)(I + A)^-1.
        This transform maps a skew-symmetric matrix A to an orthogonal matrix W.
        """
        # FIX 2: Clamp A_params to prevent ill-conditioning/instability
        A_clamped = torch.clamp(self.A_params, -self.max_A_norm, self.max_A_norm)
        A = A_clamped - A_clamped.t()
        
        I = torch.eye(self.channels, device=A.device, dtype=A.dtype)
        
        # Stability check: Ensure I + A is well-conditioned
        # Regularize to prevent singularity (I + A + epsilon * I)
        I_plus_A = I + A + 1e-6 * I
        
        # FIX: Solve (I+A)^T @ W^T = (I-A)^T  →  W = (I-A) @ (I+A)^{-1}
        W = torch.linalg.solve(I_plus_A.T, (I - A).T).T
        
        return W
    
    def get_orthogonal_matrix(self):
        """
        Retrieves the orthogonal matrix. 
        Recomputes only if training or cache is invalidated.
        """
        if not self._cache_valid or self.training:
            W = self._compute_orthogonal_matrix()
            # FIX 1: Cache the result for inference speed
            self._W_cache = W.detach().clone()
            self._cache_valid = True
            return W
        return self._W_cache
    
    def invalidate_cache(self):
        """Must be called after optimizer.step() to reflect new learning."""
        self._cache_valid = False
    
    def forward(self, x):
        """
        x: (B, C, H, W)
        Apply 1x1 Unitary Convolution.
        """
        W = self.get_orthogonal_matrix()
        
        # ELITE DEBUG: Verify orthogonality drift during training (1% frequency)
        if self.training and torch.rand(1).item() < 0.01:
            with torch.no_grad():
                I = torch.eye(self.channels, device=W.device)
                ortho_error = torch.norm(W @ W.t() - I)
                if ortho_error > 1e-4:
                    print(f"⚠️ Warning: QVS Orthogonality drift detected: {ortho_error.item():.2e}")
        
        # FIX 6: Match input dtype (fp32, fp16, bf16)
        W = W.to(x.dtype)
        W = W.view(self.channels, self.channels, 1, 1)
        return nn.functional.conv2d(x, W)
    
    def inverse(self, out):
        """
        Inverse of unitary W is simply its transpose W^T.
        """
        W = self.get_orthogonal_matrix()
        W_T = W.t().to(out.dtype).view(self.channels, self.channels, 1, 1)
        return nn.functional.conv2d(out, W_T)
    
    def check_orthogonality(self, tol=1e-3):
        """
        Verification tool: checks if W^T * W = I.
        """
        with torch.no_grad():
            W = self.get_orthogonal_matrix()
            I = torch.eye(self.channels, device=W.device)
            deviation = torch.norm(W.T @ W - I)
            is_orthogonal = deviation < tol
            return is_orthogonal, deviation.item()

def invalidate_qvs_cache(model):
    """Utility to clear QVS caches after weights are updated."""
    for m in model.modules():
        if isinstance(m, QVSUnitaryCoupling):
            m.invalidate_cache()
