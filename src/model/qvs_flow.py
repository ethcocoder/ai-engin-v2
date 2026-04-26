import torch
import torch.nn as nn

class QVSUnitaryCoupling(nn.Module):
    """
    Quantum Virtual Substrate (QVS) Unitary Coupling Layer.
    Uses Cayley parametrization to enforce orthogonal (unitary) 1x1 convolutions.
    W = (I - A)(I + A)^-1 where A is a skew-symmetric matrix.
    This preserves volume and mimics a quantum probability-conserving operation.
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # A_params are the off-diagonal elements of the skew-symmetric matrix
        self.A_params = nn.Parameter(torch.randn(channels, channels) * 0.01)

    def get_orthogonal_matrix(self):
        # Construct skew-symmetric matrix A
        # A = A_params - A_params^T
        A = self.A_params - self.A_params.t()
        
        # I is the identity matrix
        I = torch.eye(self.channels, device=A.device, dtype=A.dtype)
        
        # W = (I - A) @ (I + A)^-1
        # Since I+A is invertible for skew-symmetric A, this is well-defined
        I_plus_A_inv = torch.linalg.inv(I + A)
        W = torch.matmul(I - A, I_plus_A_inv)
        return W

    def forward(self, x):
        """
        x: (B, C, H, W)
        Modulates the hyper-latent z.
        """
        W = self.get_orthogonal_matrix() # (C, C)
        W = W.view(self.channels, self.channels, 1, 1)
        
        # Apply 1x1 convolution
        out = nn.functional.conv2d(x, W)
        return out

    def inverse(self, out):
        """
        Inverse operation. W is orthogonal, so W^-1 = W^T.
        """
        W = self.get_orthogonal_matrix()
        W_T = W.t().view(self.channels, self.channels, 1, 1)
        x = nn.functional.conv2d(out, W_T)
        return x
