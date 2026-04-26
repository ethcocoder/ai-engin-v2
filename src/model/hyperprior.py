import torch
import torch.nn as nn
from .qvs_flow import QVSUnitaryCoupling
from .quantizer import SovereignQuantizer
from .analysis import ResidualBlock

class Hyperprior(nn.Module):
    """
    Mean-Scale Hyperprior architecture with QVS unitary coupling layer.
    """
    def __init__(self, latent_dim=192, hyper_dim=128, num_components=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.hyper_dim = hyper_dim
        self.num_components = num_components
        
        # Hyper-analysis: y -> z (downsample by 2 more, so z is H/32, W/32)
        # Using stride 2 convs
        self.ha_net = nn.Sequential(
            nn.Conv2d(latent_dim, hyper_dim, 3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(hyper_dim, hyper_dim, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(hyper_dim, hyper_dim, 5, stride=2, padding=2)
        )
        
        # Hyper quantizer
        self.z_quantizer = SovereignQuantizer(hyper_dim)
        
        # QVS Unitary Coupling
        self.qvs = QVSUnitaryCoupling(hyper_dim)
        
        # Context model (Checkerboard / Channel-wise Autoregressive simplified to 3x3 masked conv)
        # For simplicity in this implementation, we use a basic 3x3 masked conv for spatial context
        # Padding is 1, so it sees causal context
        self.context_conv = MaskedConv2d(latent_dim, hyper_dim*2, kernel_size=3, padding=1)
        
        # Hyper-synthesis: z_hat -> parameters
        self.hs_net = nn.Sequential(
            nn.ConvTranspose2d(hyper_dim, hyper_dim, 5, stride=2, padding=2, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(hyper_dim, hyper_dim, 5, stride=2, padding=2, output_padding=1),
            nn.GELU(),
            nn.Conv2d(hyper_dim, hyper_dim*2, 3, stride=1, padding=1)
        )
        
        # Entropy parameters predictor
        # Inputs: features from context model + features from hyper-synthesis
        # Outputs: parameters for Gaussian mixture model (weight, mean, scale per component)
        self.param_net = nn.Sequential(
            nn.Conv2d(hyper_dim*4, hyper_dim*2, 1),
            nn.GELU(),
            nn.Conv2d(hyper_dim*2, latent_dim * num_components * 3, 1)
        )
        
    def forward(self, y, force_hard=False):
        """
        y: (B, latent_dim, H/16, W/16)
        """
        # Hyper-analysis
        z = self.ha_net(y) # (B, hyper_dim, H/32, W/32)
        
        # Quantize z
        z_hat, _ = self.z_quantizer(z, force_hard=force_hard)
        
        # QVS modulation
        z_mod = self.qvs(z_hat)
        
        # Hyper-synthesis
        hs_features = self.hs_net(z_mod) # (B, hyper_dim*2, H/16, W/16)
        
        # Context features (requires y_hat, but during training we can use y as proxy if autoregressive)
        # To avoid autoregressive decoding during training, we use quantized y
        # Actually context model must use quantized y (y_hat)
        # So we need to quantize y first or pass it in? We'll let aether_codec handle y quantization
        
        return z_hat, hs_features

    def get_gmm_params(self, hs_features, ctx_features):
        """
        hs_features: from hyper-synthesis
        ctx_features: from context model
        """
        # Combine features
        combined = torch.cat([hs_features, ctx_features], dim=1)
        
        # Predict parameters: weights, means, scales
        params = self.param_net(combined) # (B, latent_dim * 3 * num_components, H/16, W/16)
        B, _, H, W = params.shape
        params = params.view(B, self.latent_dim, self.num_components, 3, H, W)
        
        weights = params[:, :, :, 0] # (B, C, K, H, W)
        means = params[:, :, :, 1]
        scales = torch.exp(params[:, :, :, 2]) # Ensure strictly positive scale
        
        # Normalize weights
        weights = torch.softmax(weights, dim=2)
        
        return weights, means, scales


class MaskedConv2d(nn.Conv2d):
    """
    Masked convolution for causal context modeling.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', torch.zeros_like(self.weight))
        self.create_mask()

    def create_mask(self):
        k = self.kernel_size[0]
        # Type A mask: don't include center pixel
        self.mask[:, :, :k//2, :] = 1
        self.mask[:, :, k//2, :k//2] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)
