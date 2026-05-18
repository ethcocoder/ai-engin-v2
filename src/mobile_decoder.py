import torch
import torch.nn as nn

class MobileResBlock(nn.Module):
    """
    Mobile-optimized Bottleneck Residual Block using Depthwise-Separable Convolutions.
    Drastically reduces RAM bandwidth and computation overhead.
    """
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            # 1. Depthwise Convolution (groups=channels)
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            # 2. Pointwise Convolution (1x1 Linear bottleneck projection)
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.conv(x))

class MobileGenesisDecoder(nn.Module):
    """
    Low-Compute, High-Fidelity mobile reconstruction engine.
    Uses MBConv blocks and fused upsampling to execute smoothly on low-end ARM CPUs/NPUs.
    """
    def __init__(self, latent_channels: int = 16) -> None:
        super().__init__()
        # Expand latent to 128 channels (reduced from 256 for mobile weight conservation)
        self.expand = nn.Sequential(
            nn.Conv2d(latent_channels, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            MobileResBlock(128),
            MobileResBlock(128)
        )
        # Upsampling Stage 1 (128 -> PixelShuffle -> 32 channels)
        self.up1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            MobileResBlock(32)
        )
        # Upsampling Stage 2 (32 -> 128 -> PixelShuffle -> 32 channels)
        self.up2 = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            MobileResBlock(32)
        )
        # Upsampling Stage 3 (32 -> 128 -> PixelShuffle -> 32 channels)
        self.up3 = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            MobileResBlock(32)
        )
        # Final Upsampling & Synthesis (32 -> 12 -> 3 channels)
        self.up4 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.Conv2d(12, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return x
