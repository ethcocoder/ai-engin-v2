import torch
import torch.nn as nn

class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, step_size):
        # inputs: (B, C, H, W)
        # step_size: (C, 1, 1) or scalar
        outputs = torch.round(inputs / step_size) * step_size
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator (STE)
        return grad_output, None

class SovereignQuantizer(nn.Module):
    """
    Generalization of the SovereignQuantizer concept.
    Soft quantization (uniform noise) for entropy model training.
    Hard quantization (STE) for fine-tuning and inference.
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # Learnable step size per channel
        self.step_size = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x, force_hard=False):
        """
        x: (B, C, H, W)
        """
        # Ensure step_size is strictly positive
        step_size = torch.clamp(self.step_size, min=1e-4)

        if self.training and not force_hard:
            # Soft quantization: add uniform noise [-0.5, 0.5] * step_size
            noise = torch.empty_like(x).uniform_(-0.5, 0.5) * step_size
            y_hat = x + noise
        else:
            # Hard quantization: round with STE
            y_hat = STEQuantize.apply(x, step_size)

        return y_hat, step_size
