import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    """
    PatchGAN Discriminator for Adversarial Loss.
    Uses Spectral Normalization for stability.
    """
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        
        self.net = nn.Sequential(
            # Input: (B, 3, H, W)
            spectral_norm(nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (B, 64, H/2, W/2)
            spectral_norm(nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (B, 128, H/4, W/4)
            spectral_norm(nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (B, 256, H/8, W/8)
            spectral_norm(nn.Conv2d(base_channels * 4, base_channels * 8, 4, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (B, 512, H/8 - 1, W/8 - 1)
            spectral_norm(nn.Conv2d(base_channels * 8, 1, 4, stride=1, padding=1))
            # Output: (B, 1, H', W') -> Patch logits
        )

    def forward(self, x):
        return self.net(x)

class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator running PatchGAN at different scales.
    """
    def __init__(self, num_scales=2, in_channels=3):
        super().__init__()
        self.discriminators = nn.ModuleList([
            Discriminator(in_channels) for _ in range(num_scales)
        ])
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x):
        outputs = []
        for disc in self.discriminators:
            outputs.append(disc(x))
            x = self.downsample(x)
        return outputs
