import torch
import torch.nn as nn
from .attention import SwinBlock
from .analysis import ResidualBlock

class ResidualRefinementNetwork(nn.Module):
    def __init__(self, in_channels=6, out_channels=3, hidden_dim=64):
        super().__init__()
        # Input is [reconstructed, input], so 3 + 3 = 6 channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, out_channels, 3, padding=1)
        )

    def forward(self, recon, orig):
        x = torch.cat([recon, orig], dim=1)
        res = self.net(x)
        return recon + res

class SynthesisTransform(nn.Module):
    """
    Decoder: Synthesis Transform.
    Input: latent y_hat with shape (B, latent_dim, H/16, W/16)
    Output: Reconstructed image (B, 3, H, W)
    """
    def __init__(self, latent_dim=192, base_channels=128, out_channels=3):
        super().__init__()
        
        # Swin Transformer blocks at low resolution
        self.swin_blocks = nn.Sequential(
            SwinBlock(latent_dim, window_size=8, num_heads=8),
            SwinBlock(latent_dim, window_size=8, num_heads=8)
        )
        
        # Stage 1: H/8, W/8
        self.stage1 = nn.Sequential(
            ResidualBlock(latent_dim, latent_dim),
            nn.ConvTranspose2d(latent_dim, base_channels, 5, stride=2, padding=2, output_padding=1)
        )
        
        # Stage 2: H/4, W/4
        self.stage2 = nn.Sequential(
            ResidualBlock(base_channels, base_channels),
            nn.ConvTranspose2d(base_channels, base_channels, 5, stride=2, padding=2, output_padding=1)
        )
        
        # Stage 3: H/2, W/2
        self.stage3 = nn.Sequential(
            ResidualBlock(base_channels, base_channels),
            nn.ConvTranspose2d(base_channels, base_channels, 5, stride=2, padding=2, output_padding=1)
        )
        
        # Stage 4: H, W
        self.stage4 = nn.Sequential(
            ResidualBlock(base_channels, base_channels),
            nn.ConvTranspose2d(base_channels, out_channels, 5, stride=2, padding=2, output_padding=1)
        )
        
        self.rrn = ResidualRefinementNetwork(in_channels=out_channels*2, out_channels=out_channels)
        self.final_activation = nn.Tanh()

    def forward(self, y_hat, orig_input=None):
        """
        y_hat: (B, latent_dim, H/16, W/16) — quantized latent from encoder.
        orig_input: (B, 3, H, W) — original image, passed during training for RRN.
                    During inference, RRN uses the reconstructed image as its own reference.
        Returns: (B, 3, H, W) reconstructed image in range [-1, 1].
        """
        x = self.swin_blocks(y_hat)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        recon = self.stage4(x)
        
        if orig_input is not None:
            # Used for training when orig_input is available
            recon = self.rrn(recon, orig_input)
        else:
            # Inference mode, can't use original input
            recon = self.rrn(recon, recon.detach())
            
        return self.final_activation(recon)
