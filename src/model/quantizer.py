import torch
import torch.nn as nn
import torch.nn.functional as F

class DSQQuantize(torch.autograd.Function):
    """
    Differentiable Soft Quantization (DSQ).
    Hard quantization for forward pass, soft sigmoid-based gradients for backward.
    """
    @staticmethod
    def forward(ctx, y, step):
        # inputs: (B, C, H, W), step_size: (1, C, 1, 1)
        q = torch.round(y / step) * step
        
        # Soft quantization for gradients (sigmoid-based approximation)
        alpha = 10.0  # steepness
        soft_q = step * (torch.floor(y/step) + 
                 torch.sigmoid(alpha * (y/step - torch.floor(y/step) - 0.5)))
        
        ctx.save_for_backward(y, step, soft_q)
        return q

    @staticmethod
    def backward(ctx, grad_output):
        y, step, soft_q = ctx.saved_tensors
        # Compute gradient w.r.t. y using the soft approximation
        # d/dy [soft_q]
        grad_y = grad_output * torch.autograd.grad(soft_q, y, 
                    grad_outputs=torch.ones_like(soft_q), retain_graph=True)[0]
        
        # Approximate gradient for step_size (Learned Quantization)
        # d/ds [s * round(x/s)] ≈ (round(x/s) - x/s)
        q = torch.round(y / step) * step
        residual = (q - y) / (step + 1e-8)
        grad_step = (residual * grad_output).sum(dim=(0, 2, 3), keepdim=True)
        
        return grad_y, grad_step

class SovereignQuantizer(nn.Module):
    """
    Elite Sovereign Quantizer.
    Supports learned per-channel step sizes and stochastic curriculum training.
    """
    def __init__(self, channels, max_step=2.0):
        super().__init__()
        self.channels = channels
        self.max_step = max_step
        # FIX 4: Learnable step size initialized to 0.1
        self.step_size = nn.Parameter(torch.ones(channels) * 0.1)
    
    def forward(self, x, force_hard=False, hard_prob=0.0):
        """
        x: (B, C, H, W)
        force_hard: If True, uses hard quantization (deployment mode)
        hard_prob: Probability of using hard quantization (training curriculum)
        """
        # FIX 4: Clamp step size to prevent information destruction
        step_size = torch.clamp(self.step_size, min=1e-4, max=self.max_step)
        # Standardize broadcasting
        step_expanded = step_size.view(1, -1, 1, 1)
        
        # FIX 5: Stochastic Hard Quantization (Curriculum)
        use_hard = force_hard
        if self.training and not force_hard:
            # Gradually transition from soft noise to hard integers
            use_hard = torch.rand(1).item() < hard_prob
        
        if use_hard:
            # ELITE FIX: Hard quantization with DSQ (Soft Grad)
            y_hat = DSQQuantize.apply(x, step_expanded)
        else:
            # Soft quantization: uniform noise over the quantization bin
            # This models the "average" effect of quantization for better gradient flow
            noise = torch.empty_like(x).uniform_(-0.5, 0.5) * step_expanded
            y_hat = x + noise
        
        return y_hat, step_size
