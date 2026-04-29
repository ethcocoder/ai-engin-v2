import torch
import torch.nn as nn
import math
from .attention import SwinBlock

class BlurPoolDownsample(nn.Module):
    """
    Anti-aliased downsampling with blur-then-stride.
    Prevents aliasing artifacts during the encoding stage.
    """
    def __init__(self, in_ch, out_ch, kernel_size=5):
        super().__init__()
        # FIX 2: Fixed average pool for anti-aliasing
        self.blur = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=2, padding=kernel_size//2)
        self.norm = nn.GroupNorm(min(32, out_ch//4), out_ch)
        self.act = nn.GELU()
    
    def forward(self, x):
        x = self.blur(x)
        return self.act(self.norm(self.conv(x)))

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, drop_path_rate=0.0):
        super().__init__()
        # FIX 1: Add GroupNorm
        self.norm0 = nn.GroupNorm(min(32, in_ch//4), in_ch) if in_ch > 4 else nn.Identity()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(32, out_ch//4), out_ch)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, out_ch//4), out_ch)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.drop_path_rate = drop_path_rate
    
    def forward(self, x):
        res = self.shortcut(x)
        out = self.norm0(x)
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.norm2(out)
        
        # FIX 4: Stochastic Depth (Drop Path)
        if self.training and torch.rand(1).item() < self.drop_path_rate:
            return res
        
        # FIX 5: Post-activation stability
        return self.act(out + res)

class PositionalEncoding2D(nn.Module):
    """
    FIX 8: 2D Sinusoidal Positional Encoding for Transformer blocks.
    """
    def __init__(self, channels, max_h=64, max_w=64):
        super().__init__()
        pe = torch.zeros(1, channels, max_h, max_w)
        y_pos = torch.arange(max_h).unsqueeze(1).float()
        x_pos = torch.arange(max_w).unsqueeze(0).float()
        div_term = torch.exp(torch.arange(0, channels, 2).float() * 
                           -(math.log(10000.0) / channels))
        
        for i in range(0, channels, 2):
            pe[0, i, :, :] = torch.sin(y_pos * div_term[i//2]) + \
                            torch.sin(x_pos * div_term[i//2])
            if i+1 < channels:
                pe[0, i+1, :, :] = torch.cos(y_pos * div_term[i//2]) + \
                                  torch.cos(x_pos * div_term[i//2])
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :, :x.size(2), :x.size(3)]

class AnalysisTransform(nn.Module):
    """
    Encoder: Anti-Aliased, Attention-Augmented Analysis Transform.
    """
    def __init__(self, in_channels=3, latent_dim=192, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        
        # FIX 6: Progressive channel growth
        self.stage1 = nn.Sequential(
            BlurPoolDownsample(in_channels, 64, 5),
            ResidualBlock(64, 64, drop_path_rate=0.0)
        )
        self.stage2 = nn.Sequential(
            BlurPoolDownsample(64, 128, 5),
            ResidualBlock(128, 128, drop_path_rate=0.05)
        )
        self.stage3 = nn.Sequential(
            BlurPoolDownsample(128, 256, 5),
            ResidualBlock(256, 256, drop_path_rate=0.1)
        )
        self.stage4 = nn.Sequential(
            BlurPoolDownsample(256, latent_dim, 5),
            ResidualBlock(latent_dim, latent_dim, drop_path_rate=0.1)
        )
        
        # ELITE ACCURACY: Global Context Path
        # Captures image-wide statistics to inform the local latents
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            nn.Conv2d(latent_dim, latent_dim, 1),
            nn.GELU(),
            nn.Conv2d(latent_dim, latent_dim, 1),
            nn.Sigmoid()
        )
        
        # FIX 8: Positional encoding
        self.pos_enc = PositionalEncoding2D(latent_dim)
        
        if self.use_attention:
            self.swin_blocks = nn.Sequential(
                SwinBlock(latent_dim, window_size=8, num_heads=8),
                SwinBlock(latent_dim, window_size=8, num_heads=8)
            )
        else:
            self.swin_blocks = nn.Identity()
        
        # FIX 9: Explicit weight init
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # FIX 9: Explicit weight init (GELU -> RELU for gain calc)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x, return_skips=None):
        # FIX 10: Automatic inference skip handling
        if return_skips is None:
            return_skips = self.training
        
        skips = []
        
        x = self.stage1(x); skips.append(x)  # H/2, 64ch
        x = self.stage2(x); skips.append(x)  # H/4, 128ch
        x = self.stage3(x); skips.append(x)  # H/8, 256ch
        
        x = self.stage4(x)
        
        # ELITE ACCURACY: Apply Global Context Gating
        g_context = self.global_context(x)
        g_context = F.interpolate(g_context, size=x.shape[-2:], mode='bilinear')
        x = x * g_context
        
        # FIX 3: Swin output not in skips fix: add H/16 skip before attention
        skips.append(x)  # H/16, 192ch
        
        x = self.pos_enc(x)
        x = self.swin_blocks(x)
        
        if return_skips:
            # FIX 7: Reverse order for decoder [H/16, H/8, H/4, H/2]
            return x, skips[::-1]
        return x, None
