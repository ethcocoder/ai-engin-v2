import torch
import torch.nn as nn
from .attention import SwinBlock

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        return out + res

class AnalysisTransform(nn.Module):
    """
    Encoder: Analysis Transform.
    Input: (B, 3, H, W)
    Output: latent y with shape (B, latent_dim, H/16, W/16)
    """
    def __init__(self, in_channels=3, base_channels=128, latent_dim=192):
        super().__init__()
        
        # Stage 1: H/2, W/2
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 5, stride=2, padding=2),
            ResidualBlock(base_channels, base_channels)
        )
        
        # Stage 2: H/4, W/4
        self.stage2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 5, stride=2, padding=2),
            ResidualBlock(base_channels, base_channels)
        )
        
        # Stage 3: H/8, W/8
        self.stage3 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 5, stride=2, padding=2),
            ResidualBlock(base_channels, base_channels)
        )
        
        # Stage 4: H/16, W/16
        self.stage4 = nn.Sequential(
            nn.Conv2d(base_channels, latent_dim, 5, stride=2, padding=2),
            ResidualBlock(latent_dim, latent_dim)
        )
        
        # Swin Transformer blocks at lowest resolution
        self.swin_blocks = nn.Sequential(
            SwinBlock(latent_dim, window_size=8, num_heads=8),
            SwinBlock(latent_dim, window_size=8, num_heads=8)
        )

    def forward(self, x):
        """
        x: (B, 3, H, W)
        Returns: (B, latent_dim, H/16, W/16)
        """
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.swin_blocks(x)
        return x
