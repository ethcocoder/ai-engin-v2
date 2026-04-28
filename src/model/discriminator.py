import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class BlurPool(nn.Module):
    """
    Anti-aliased downsampling for multi-scale discriminator.
    Ensures gradients are consistent across different resolutions.
    """
    def __init__(self):
        super().__init__()
        self.blur = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.pool = nn.AvgPool2d(2, stride=2)
    
    def forward(self, x):
        return self.pool(self.blur(x))

class Discriminator(nn.Module):
    """
    Elite PatchGAN Discriminator with Spectral Norm and InstanceNorm.
    Supports Feature Matching extraction for stabilized training.
    """
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        
        # FIX 1, 3, 4: Added InstanceNorm, removed bias, removed inplace LeakyReLU
        self.layers = nn.ModuleList([
            # Layer 0: H/2
            nn.Sequential(
                spectral_norm(nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1, bias=False)),
                nn.LeakyReLU(0.2, inplace=False)
            ),
            # Layer 1: H/4
            nn.Sequential(
                spectral_norm(nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1, bias=False)),
                nn.InstanceNorm2d(base_channels * 2),
                nn.LeakyReLU(0.2, inplace=False)
            ),
            # Layer 2: H/8
            nn.Sequential(
                spectral_norm(nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1, bias=False)),
                nn.InstanceNorm2d(base_channels * 4),
                nn.LeakyReLU(0.2, inplace=False)
            ),
            # Layer 3: H/8 (Stride 1)
            nn.Sequential(
                spectral_norm(nn.Conv2d(base_channels * 4, base_channels * 8, 4, stride=1, padding=1, bias=False)),
                nn.InstanceNorm2d(base_channels * 8),
                nn.LeakyReLU(0.2, inplace=False)
            ),
        ])
        
        # Final layer: Outputs Logits (FIX 2)
        self.final = spectral_norm(nn.Conv2d(base_channels * 8, 1, 4, stride=1, padding=1, bias=False))
    
    def forward(self, x, return_features=False):
        features = []
        for layer in self.layers:
            x = layer(x)
            if return_features:
                features.append(x)
        
        x = self.final(x)
        
        if return_features:
            return x, features
        return x

class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale PatchGAN (standard for high-resolution images like 512x512).
    """
    def __init__(self, num_scales=3, in_channels=3):
        super().__init__()
        # FIX 7: Default to 3 scales for elite performance
        self.discriminators = nn.ModuleList([
            Discriminator(in_channels) for _ in range(num_scales)
        ])
        self.downsample = BlurPool()
    
    def forward(self, x, return_features=False):
        outputs = []
        all_features = [] if return_features else None
        
        for disc in self.discriminators:
            if return_features:
                out, feats = disc(x, return_features=True)
                outputs.append(out)
                all_features.append(feats)
            else:
                outputs.append(disc(x))
            x = self.downsample(x)
        
        if return_features:
            return outputs, all_features
        return outputs
