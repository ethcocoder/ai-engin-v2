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

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, y):
        """
        y: (B, C, H, W) unquantized latent representation.
        CRITICAL: This entire method runs in FP32. Autocast's FP16
        silently converts bmm back to half precision, causing overflow -> NaN.
        """
        B, C, H, W = y.shape
        y_flat = y.float().view(B, C, -1)  # Force FP32
        
        # Reduced density matrix: rho[b]_ij = (1/N) sum_k y[b,i,k] * y[b,j,k]
        rho = torch.bmm(y_flat, y_flat.transpose(1, 2))  # (B, C, C)
        
        # Normalize trace to 1 (making it a valid density matrix)
        trace = torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1) + 1e-7
        rho = rho / trace
        
        if self.use_renyi2:
            # Renyi-2 entropy: S_2 = -log2(Tr(rho^2))
            rho_sq = torch.bmm(rho, rho)  # (B, C, C)
            purity = torch.diagonal(rho_sq, dim1=-2, dim2=-1).sum(-1)  # Tr(rho^2)
            purity = torch.clamp(purity, min=1e-9, max=1.0)
            entropy = -torch.log2(purity)
        else:
            eigenvalues = torch.linalg.eigvalsh(rho)  # (B, C)
            eigenvalues = torch.clamp(eigenvalues, min=1e-9)
            entropy = -torch.sum(eigenvalues * torch.log2(eigenvalues), dim=-1)
        
        max_entropy = torch.log2(torch.tensor(C, dtype=torch.float32, device=y.device))
        norm_entropy = entropy / (max_entropy + 1e-9)
        
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
