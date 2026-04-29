import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """
    Refined Channel Attention with learned temperature.
    Ensures stable gating of feature channels.
    """
    def __init__(self, in_planes):
        super().__init__()
        # FIX 5: Balanced bottleneck ratio
        ratio = max(4, in_planes // 64)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        # FIX 5: Standard sigmoid with temperature
        return torch.sigmoid((avg_out + max_out) / self.temperature)

class MultiScaleSpatialAttention(nn.Module):
    """
    Spatially-aware attention hierarchy.
    """
    def __init__(self, channels):
        super().__init__()
        self.local = nn.Conv2d(2, 1, 7, padding=3)
        # FIX 6: Global path is now spatially aware via 1x1 projection
        self.global_proj = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, 1, 1)
        )

    def forward(self, x):
        # Local path: focuses on detail
        avg = x.mean(1, keepdim=True)
        mx = x.max(1, keepdim=True)[0]
        local_att = torch.sigmoid(self.local(torch.cat([avg, mx], 1)))
        
        # Global path: focuses on semantic regions
        global_att = torch.sigmoid(self.global_proj(x))
        
        return local_att * global_att

class EliteRefinementBlock(nn.Module):
    """
    The core of the synthesis engine. 
    Combines convolution with dual-attention refinement.
    """
    def __init__(self, channels, drop_rate=0.1):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = MultiScaleSpatialAttention(channels)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(min(32, channels//4), channels),
            nn.GELU(),
            nn.Dropout2d(drop_rate), # FIX 8: Regularization
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        residual = x
        # FIX 4: CBAM order: Conv -> Refine
        out = self.conv(x)
        out = out * self.ca(out)
        out = out * self.sa(out)
        return out + residual

class ResidualRefinementNetwork(nn.Module):
    """
    Honest refinement engine. 
    Adds fine-frequency detail to the base reconstruction.
    ELITE ACCURACY: Deeper refinement with multi-scale feature injection.
    """
    def __init__(self, in_channels=3, feature_channels=32, hidden=64):
        super().__init__()
        self.initial = nn.Conv2d(in_channels + feature_channels, hidden, 3, padding=1)
        
        # ELITE ACCURACY: 3 blocks instead of 2 for better detail
        self.refine_blocks = nn.Sequential(
            EliteRefinementBlock(hidden),
            EliteRefinementBlock(hidden),
            EliteRefinementBlock(hidden)
        )
        
        self.final = nn.Conv2d(hidden, in_channels, 3, padding=1)
        # FIX 7: Start weak so the decoder body learns first
        self.residual_gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, recon, decoder_features):
        x = torch.cat([recon, decoder_features], dim=1)
        x = self.initial(x)
        x = self.refine_blocks(x)
        res = self.final(x)
        gate = torch.sigmoid(self.residual_gate)
        # Clamp residual to prevent color explosion
        return recon + gate * torch.clamp(res, -0.5, 0.5)

class SynthesisTransform(nn.Module):
    """
    The Elite Decoder Body.
    Implements U-Net skip fusion and hierarchical upsampling.
    """
    def __init__(self, latent_channels=192, hidden_channels=[512, 256, 128, 64, 32], out_channels=3):
        super().__init__()
        
        # Aligned with AnalysisTransform channels: [H/16, H/8, H/4, H/2]
        self.skip_channels = [192, 256, 128, 64]
        
        # Feature extractor (FIX 9: Deep extraction from latent)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_channels[0], 3, padding=1),
            nn.GroupNorm(32, hidden_channels[0]),
            nn.GELU(),
            nn.Conv2d(hidden_channels[0], hidden_channels[0], 3, padding=1),
            nn.GroupNorm(32, hidden_channels[0]),
            nn.GELU(),
        )
        
        # Fusion for the deepest scale (H/16)
        self.latent_skip_fusion = nn.Conv2d(
            hidden_channels[0] + self.skip_channels[0], hidden_channels[0], 1
        )
        
        self.upsample_stages = nn.ModuleList()
        self.skip_fusions = nn.ModuleList()
        
        for i in range(len(hidden_channels) - 1):
            ch_in = hidden_channels[i]
            ch_out = hidden_channels[i+1]
            
            # Learned upsampling (Resize-Conv)
            self.upsample_stages.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(ch_in, ch_out, 3, padding=1),
                nn.GroupNorm(min(32, ch_out//4), ch_out),
                nn.GELU(),
                EliteRefinementBlock(ch_out, drop_rate=0.1)
            ))
            
            # Fuse skips from encoder [H/8, H/4, H/2]
            if i + 1 < len(self.skip_channels):
                self.skip_fusions.append(nn.Conv2d(
                    ch_out + self.skip_channels[i+1], ch_out, 1
                ))
            else:
                # No skip for the final upsample (H/2 -> H)
                self.skip_fusions.append(nn.Identity())
        
        # FIX 3: Tanh for smooth bounded colors
        self.to_rgb = nn.Sequential(
            nn.Conv2d(hidden_channels[-1], out_channels, 3, padding=1),
            nn.Tanh()
        )
        
        self.rrn = ResidualRefinementNetwork(
            in_channels=out_channels, feature_channels=hidden_channels[-1],
            hidden=64 # Reduced from 128 to prevent OOM at 512x512
        )
    
    def forward(self, y_hat, encoder_skips=None):
        x = self.feature_extractor(y_hat)
        
        # Stage 0 Fusion: H/16
        if encoder_skips is not None:
            skip16 = encoder_skips[0]
            if x.shape[-2:] != skip16.shape[-2:]:
                skip16 = F.interpolate(skip16, size=x.shape[-2:], mode='bilinear')
            x = torch.cat([x, skip16], dim=1)
            x = self.latent_skip_fusion(x)
        
        # Stage 1-3 Fusion: H/8, H/4, H/2
        for i, stage in enumerate(self.upsample_stages):
            x = stage(x)
            
            if encoder_skips is not None:
                skip_idx = i + 1
                if skip_idx < len(encoder_skips):
                    skip = encoder_skips[skip_idx]
                    if x.shape[-2:] != skip.shape[-2:]:
                        skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear')
                    x = torch.cat([x, skip], dim=1)
                    x = self.skip_fusions[i](x)
        
        # Final lightweight features for refinement
        refinement_features = x # (B, 32, 512, 512)
        
        # Base reconstruction
        recon = self.to_rgb(x)
        
        # Final Honest Refinement using lightweight features
        final = self.rrn(recon, refinement_features)
        
        # Final safety clamp
        return torch.clamp(final, -1.0, 1.0)
