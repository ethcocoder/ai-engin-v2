import torch
import torch.nn as nn

class EntanglementRegularizer(nn.Module):
    """
    Computes von Neumann entropy of the latent representation's reduced density matrix.
    Acts as a learned sparsity/disentanglement prior to preserve the quantum branding.
    """
    def __init__(self, weight=0.001):
        super().__init__()
        self.weight = weight

    def forward(self, y):
        """
        y: (B, C, H, W) latent representation.
        """
        B, C, H, W = y.shape
        # Flatten spatial dimensions: (B, C, N)
        y_flat = y.view(B, C, -1)
        
        # Compute covariance matrix: (B, C, C)
        cov = torch.bmm(y_flat, y_flat.transpose(1, 2))
        
        # Normalize to create a valid density matrix (trace = 1)
        # Add small epsilon to avoid division by zero
        trace = torch.diagonal(cov, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1) + 1e-9
        rho = cov / trace
        
        # Compute eigenvalues (since rho is symmetric positive semi-definite)
        # Using eigh for symmetric matrices
        eigenvalues = torch.linalg.eigvalsh(rho)
        
        # Clip eigenvalues to avoid log(0)
        eigenvalues = torch.clamp(eigenvalues, min=1e-9)
        
        # von Neumann entropy: S(rho) = -Tr(rho log rho) = -sum(lambda * log(lambda))
        entropy = -torch.sum(eigenvalues * torch.log(eigenvalues), dim=-1)
        
        # Average over batch
        loss = entropy.mean()
        
        return loss * self.weight
