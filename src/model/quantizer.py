import torch
import torch.nn as nn
import torch.nn.functional as F

class STEQuantize(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) for Hard Quantization.
    Allows gradients to flow through the non-differentiable round() function.
    """
    @staticmethod
    def forward(ctx, inputs, step_size):
        # inputs: (B, C, H, W), step_size: (1, C, 1, 1)
        outputs = torch.round(inputs / step_size) * step_size
        ctx.save_for_backward(inputs, step_size, outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        inputs, step_size, outputs = ctx.saved_tensors
        # Gradient w.r.t. inputs: Identity (Straight-Through)
        grad_input = grad_output
        
        # FIX 1: Approximate gradient for step_size (Learned Quantization)
        # d/ds [s * round(x/s)] ≈ (round(x/s) - x/s) 
        residual = (outputs - inputs) / (step_size + 1e-8)
        grad_step = (residual * grad_output).sum(dim=(0, 2, 3), keepdim=True)
        
        # Reshape to match Parameter shape (C,)
        return grad_input, grad_step.view(-1)

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
            # Hard quantization with STE
            # During hard mode, we use detach on step to maintain stability if needed, 
            # but here we allow learning.
            y_hat = STEQuantize.apply(x, step_expanded)
        else:
            # Soft quantization: uniform noise over the quantization bin
            # This models the "average" effect of quantization for better gradient flow
            noise = torch.empty_like(x).uniform_(-0.5, 0.5) * step_expanded
            y_hat = x + noise
        
        return y_hat, step_size
