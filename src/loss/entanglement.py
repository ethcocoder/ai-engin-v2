import torch
import torch.nn as nn

class EntanglementRegularizer(nn.Module):
    """
    Elite Differentiable Channel Regularizer.
    Uses soft histogram entropy (RBF kernel) to maximize channel independence.
    """
    def __init__(self, num_bins=256, temperature=0.5, weight=0.01):
        super().__init__()
        self.weight = weight
        self.temperature = temperature
        self.register_buffer('bins', torch.linspace(-10, 10, num_bins))
    
    def forward(self, y_hat):
        # 1. Flatten to (N, 1)
        y_flat = y_hat.float().reshape(-1, 1)
        bins = self.bins.view(1, -1)
        
        # 2. Soft histogram via RBF kernel (fully differentiable)
        # Compute distances: -((y_flat - bins) ** 2) / (2 * temperature ** 2)
        distances = -((y_flat - bins) ** 2) / (2 * self.temperature ** 2)
        weights = torch.softmax(distances, dim=1)  # Soft assignment to bins
        
        # 3. Normalize to probability distribution
        hist = weights.mean(dim=0)
        hist = hist / (hist.sum() + 1e-10)
        
        # 4. Shannon entropy (maximize = minimize redundancy)
        entropy = -(hist * torch.log(hist + 1e-10)).sum()
        
        # Penalize low entropy (peaked distributions = inefficient coding)
        return -entropy * self.weight

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
