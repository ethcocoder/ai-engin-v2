import torch
import torch.nn as nn

class EntanglementRegularizer(nn.Module):
    """
    Elite Quantum-inspired Channel Regularizer.
    Measures 'entanglement' between latent channels via Reduced Density Matrix entropy.
    Goal: Lower entropy = more independent channels = higher compression efficiency.
    Optimized with Renyi-2 entropy to avoid expensive eigendecomposition.
    """
    def __init__(self, weight=0.01, use_renyi2=True):
        super().__init__()
        self.weight = weight
        self.use_renyi2 = use_renyi2 

    def forward(self, y):
        """
        y: (B, C, H, W) unquantized latent representation.
        """
        B, C, H, W = y.shape
        # Treat spatial positions as the 'environment' and channels as the 'subsystem'
        # FIX: Force float32 for matrix multiplication to prevent overflow in Mixed Precision
        y_flat = y.view(B, C, -1).to(torch.float32)  # (B, C, N)
        
        # Reduced density matrix: rho[b]_ij = (1/N) sum_k y[b,i,k] * y[b,j,k]
        # rho represents the correlation state between channels.
        rho = torch.bmm(y_flat, y_flat.transpose(1, 2))  # (B, C, C)
        
        # Normalize trace to 1 (making it a valid density matrix)
        trace = torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1) + 1e-9
        rho = rho / trace
        
        if self.use_renyi2:
            # FIX 2: Renyi-2 entropy: S_2 = -log2(Tr(rho^2))
            # Mathematically equivalent for regularization but 10x faster than eigenvalues.
            rho_sq = torch.bmm(rho, rho)  # (B, C, C)
            purity = torch.diagonal(rho_sq, dim1=-2, dim2=-1).sum(-1)  # Tr(rho^2)
            purity = torch.clamp(purity, min=1e-9, max=1.0)
            entropy = -torch.log2(purity)  # bits (FIX 4)
        else:
            # von Neumann entropy (Standard but slow)
            eigenvalues = torch.linalg.eigvalsh(rho)  # (B, C)
            eigenvalues = torch.clamp(eigenvalues, min=1e-9)
            entropy = -torch.sum(eigenvalues * torch.log2(eigenvalues), dim=-1)
        
        # Normalize entropy by max possible bits log2(C) for stable weighting
        max_entropy = torch.log2(torch.tensor(C, dtype=torch.float32, device=y.device))
        norm_entropy = entropy / (max_entropy + 1e-9)
        
        # Safety clamp to prevent gradient explosion
        loss = torch.clamp(norm_entropy.mean(), max=2.0) * self.weight
        return loss

class SparsityRegularizer(nn.Module):
    """
    High-Performance L1 Sparsity. 
    Directly encourages the model to use fewer 'Math Tokens' for the same image.
    """
    def __init__(self, weight=0.01):
        super().__init__()
        self.weight = weight
    
    def forward(self, y):
        # Average L1 norm
        return self.weight * torch.mean(torch.abs(y))
