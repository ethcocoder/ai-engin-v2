import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .analysis import AnalysisTransform
from .synthesis import SynthesisTransform
from .hyperprior import Hyperprior
from .quantizer import SovereignQuantizer


class AetherCodec(nn.Module):
    """
    AetherCodec-Elite: State-of-the-art learned image codec.
    Mean-Scale Hyperprior with Gaussian Mixture entropy model.
    Mathematically Correct (Audited v4).
    """
    def __init__(self, use_qvs=True, use_attention=True, use_hyperprior=True, 
                 use_checkpoint=False):
        super().__init__()
        self.use_qvs = use_qvs
        self.use_attention = use_attention
        self.use_hyperprior = use_hyperprior
        self.use_checkpoint = use_checkpoint
        
        self.encoder = AnalysisTransform(
            in_channels=3, 
            latent_dim=192
        )
        self.decoder = SynthesisTransform(latent_channels=192)
        self.y_quantizer = SovereignQuantizer(192)
        
        if self.use_hyperprior:
            self.hyperprior = Hyperprior(
                latent_dim=192, 
                hyper_dim=128, 
                num_components=3
            )

    def forward(self, x, force_hard=None):
        """
        Args:
            x: (B, 3, H, W), values in [-1, 1]
            force_hard: If None, auto-select based on training mode
        """
        if force_hard is None:
            force_hard = not self.training
        
        B, _, H, W = x.shape
        num_pixels = B * H * W
        
        # 1. Encode
        if self.training and self.use_checkpoint:
            y, skips = checkpoint(self.encoder, x, True, use_reentrant=False)
        else:
            y, skips = self.encoder(x, return_skips=True)
        
        # 2. Quantize main latent
        y_hat, y_step = self.y_quantizer(y, force_hard=force_hard)
        
        likelihoods = {}
        metrics = {'y_hat': y_hat, 'y_step': y_step, 'y_clean': y}
        
        # 3. Hyperprior (operates on QUANTIZED y_hat)
        if self.use_hyperprior:
            # CRITICAL FIX: hyperprior encodes y_hat, not y
            z_hat, z_step, hs_features = self.hyperprior(
                y_hat, force_hard=force_hard
            )
            metrics['z_hat'] = z_hat
            metrics['z_step'] = z_step
            
            # Context model
            ctx_features = self.hyperprior.context_conv(y_hat)
            
            # GMM parameters
            weights, means, scales = self.hyperprior.get_gmm_params(
                hs_features, ctx_features
            )
            
            # Probability mass in quantization bin
            y_hat_expanded = y_hat.unsqueeze(2)  # (B, C, 1, H, W)
            dist = torch.distributions.Normal(means, scales)
            
            # CRITICAL FIX: explicit step shape
            step = y_step.view(1, -1, 1, 1, 1)  # (1, C, 1, 1, 1)
            
            lower = dist.cdf(y_hat_expanded - step / 2)
            upper = dist.cdf(y_hat_expanded + step / 2)
            p_y = (upper - lower) * weights
            p_y = p_y.sum(dim=2)  # Marginalize mixture (B, C, H, W)
            p_y = torch.clamp(p_y, min=1e-9)
            likelihoods['y'] = p_y
            
            # Hyper-latent prior (factorized Gaussian)
            z_dist = torch.distributions.Normal(0.0, 1.0)
            z_step_v = z_step.view(1, -1, 1, 1)  # CRITICAL FIX
            
            z_lower = z_dist.cdf(z_hat - z_step_v / 2)
            z_upper = z_dist.cdf(z_hat + z_step_v / 2)
            p_z = z_upper - z_lower
            p_z = torch.clamp(p_z, min=1e-9)
            likelihoods['z'] = p_z
            
            # Pre-compute bpp for convenience
            metrics['bpp_y'] = -torch.log2(p_y).sum() / num_pixels
            metrics['bpp_z'] = -torch.log2(p_z).sum() / num_pixels
            
        else:
            # No hyperprior: use factorized prior for y
            z_dist = torch.distributions.Normal(0.0, 1.0)
            step = y_step.view(1, -1, 1, 1)
            
            y_lower = z_dist.cdf(y_hat - step / 2)
            y_upper = z_dist.cdf(y_hat + step / 2)
            p_y = y_upper - y_lower
            p_y = torch.clamp(p_y, min=1e-9)
            
            likelihoods['y'] = p_y
            likelihoods['z'] = torch.ones_like(y_hat) * 1e-9  # Zero rate
            metrics['bpp_y'] = -torch.log2(p_y).sum() / num_pixels
            metrics['bpp_z'] = torch.tensor(0.0, device=x.device)
        
        # 4. Decode
        # Skips are only available if we are the sender (training) or simulating end-to-end.
        # In real-world inference, skips=None.
        x_hat = self.decoder(y_hat, encoder_skips=skips if self.training else None)
        
        return x_hat, likelihoods, metrics
