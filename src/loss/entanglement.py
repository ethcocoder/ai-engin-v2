import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

class EntanglementRegularizer(nn.Module):
    """
    Differentiable entropy regularization via RBF soft histogram.
    Replaces deprecated Rényi-2 entropy with stable Shannon entropy.
    """
    
    def __init__(self, weight=0.01, num_bins=256, temperature=0.5, **kwargs):
        super().__init__()
        self.weight = weight
        self.num_bins = num_bins
        self.temperature = temperature
        
        # Non-trainable histogram bins
        self.register_buffer('bins', torch.linspace(-10, 10, num_bins))
        
        # Backward compatibility: warn about deprecated args
        if kwargs:
            warnings.warn(
                f"EntanglementRegularizer: deprecated arguments ignored: {list(kwargs.keys())}. "
                f"RBF soft histogram is now always used.", 
                DeprecationWarning, 
                stacklevel=2
            )
    
    def forward(self, y_hat):
        """
        Args:
            y_hat: Latent tensor (B, C, H, W)
        Returns:
            Scalar loss: negative entropy (minimize = maximize entropy)
        """
        # Flatten to (N, 1)
        y_flat = y_hat.float().reshape(-1, 1)
        bins = self.bins.view(1, -1)
        
        # RBF kernel for soft histogram
        distances = -((y_flat - bins) ** 2) / (2 * self.temperature ** 2)
        weights = F.softmax(distances, dim=1)
        
        # Build probability distribution
        hist = weights.mean(dim=0)
        hist = hist / (hist.sum() + 1e-10)
        
        # Shannon entropy
        entropy = -(hist * torch.log(hist + 1e-10)).sum()
        
        # Return negative entropy as loss (we want to maximize entropy)
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
