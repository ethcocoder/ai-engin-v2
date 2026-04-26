import torch
import torch.nn as nn
import torch.nn.functional as F

from .analysis import AnalysisTransform
from .synthesis import SynthesisTransform
from .hyperprior import Hyperprior
from .quantizer import SovereignQuantizer

class AetherCodec(nn.Module):
    """
    AetherCodec-Elite: State-of-the-art learned image codec.
    Incorporates Swin Transformers, QVS, and Mean-Scale Hyperprior.
    """
    def __init__(self, use_qvs=True, use_attention=True, use_hyperprior=True):
        super().__init__()
        self.use_qvs = use_qvs
        self.use_attention = use_attention
        self.use_hyperprior = use_hyperprior
        
        self.encoder = AnalysisTransform(in_channels=3, base_channels=128, latent_dim=192)
        self.decoder = SynthesisTransform(latent_dim=192, base_channels=128, out_channels=3)
        self.y_quantizer = SovereignQuantizer(192)
        
        if self.use_hyperprior:
            self.hyperprior = Hyperprior(latent_dim=192, hyper_dim=128, num_components=3)

    def forward(self, x, force_hard=False):
        """
        x: (B, 3, H, W)
        """
        # 1. Encode
        y = self.encoder(x) # (B, 192, H/16, W/16)
        
        # 2. Quantize latent
        y_hat, y_step = self.y_quantizer(y, force_hard=force_hard)
        
        likelihoods = {}
        
        # 3. Hyperprior and Entropy Model
        if self.use_hyperprior:
            z_hat, hs_features = self.hyperprior(y, force_hard=force_hard)
            
            # Context model uses y_hat for spatial context
            ctx_features = self.hyperprior.context_conv(y_hat)
            
            # Predict parameters
            weights, means, scales = self.hyperprior.get_gmm_params(hs_features, ctx_features)
            
            # Calculate likelihoods of y_hat under the GMM
            # p(y_hat) = sum_i w_i * N(y_hat; mu_i, sigma_i^2)
            # To compute rate, we integrate over the quantization bin, 
            # simplified as p(y_hat) * step_size for continuous approximation
            
            y_hat_expanded = y_hat.unsqueeze(2) # (B, C, 1, H, W)
            means = means # (B, C, K, H, W)
            scales = scales # (B, C, K, H, W)
            weights = weights # (B, C, K, H, W)
            
            # Gaussian PDF
            dist = torch.distributions.Normal(means, scales)
            
            # Probability mass in the bin [y_hat - step/2, y_hat + step/2]
            # using CDF for numerical stability
            step = y_step.unsqueeze(2) # (1, C, 1, 1, 1)
            lower = dist.cdf(y_hat_expanded - step / 2)
            upper = dist.cdf(y_hat_expanded + step / 2)
            p_y = (upper - lower) * weights
            p_y = p_y.sum(dim=2) # Marginalize over K components (B, C, H, W)
            
            # Clamp to avoid log(0)
            p_y = torch.clamp(p_y, min=1e-9)
            likelihoods['y'] = p_y
            
            # For z_hat, use a simple non-parametric factorized prior or a fixed logistic/gaussian
            # Here we use a fixed standard normal for z for simplicity
            z_dist = torch.distributions.Normal(0.0, 1.0)
            z_lower = z_dist.cdf(z_hat - 0.5)
            z_upper = z_dist.cdf(z_hat + 0.5)
            p_z = z_upper - z_lower
            p_z = torch.clamp(p_z, min=1e-9)
            likelihoods['z'] = p_z
        else:
            # Simple factorized prior if no hyperprior
            z_dist = torch.distributions.Normal(0.0, 1.0)
            y_step_expanded = y_step
            y_lower = z_dist.cdf(y_hat - y_step_expanded / 2)
            y_upper = z_dist.cdf(y_hat + y_step_expanded / 2)
            p_y = y_upper - y_lower
            p_y = torch.clamp(p_y, min=1e-9)
            likelihoods['y'] = p_y
            
        # 4. Decode
        x_hat = self.decoder(y_hat, orig_input=x)
        
        return x_hat, likelihoods, y
