import torch
import torch.nn as nn
import torch.nn.functional as F

class DSQQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, step):
        q = torch.round(y / step) * step
        ctx.save_for_backward(y, step)
        return q

    @staticmethod
    def backward(ctx, grad_output):
        y, step = ctx.saved_tensors
        alpha = 10.0
        relative_y = y / step
        shifted_y = relative_y - torch.floor(relative_y) - 0.5
        sig = torch.sigmoid(alpha * shifted_y)
        grad_soft = alpha * sig * (1.0 - sig)
        grad_y = grad_output * grad_soft
        q = torch.round(relative_y) * step
        residual = (q - y) / (step + 1e-8)
        grad_step = (residual * grad_output).sum(dim=(0, 2, 3), keepdim=True)
        return grad_y, grad_step

class SovereignQuantizer(nn.Module):
    def __init__(self, channels, max_step=2.0):
        super().__init__()
        self.channels = channels
        self.max_step = max_step
        self.step_size = nn.Parameter(torch.ones(channels) * 0.1)
    
    def forward(self, x, force_hard=False, hard_prob=0.0):
        step_size = torch.clamp(self.step_size, min=1e-4, max=self.max_step)
        step_expanded = step_size.view(1, -1, 1, 1)
        use_hard = force_hard
        if self.training and not force_hard:
            use_hard = torch.rand(1).item() < hard_prob
        if use_hard:
            y_hat = DSQQuantize.apply(x, step_expanded)
        else:
            noise = torch.empty_like(x).uniform_(-0.5, 0.5) * step_expanded
            y_hat = x + noise
        return y_hat, step_size
