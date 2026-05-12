import torch
import torch.nn as nn
import torch.nn.functional as F
from .qvs_flow import QVSUnitaryCoupling
from .quantizer import SovereignQuantizer

class MaskedConv2d(nn.Conv2d):
    """
    Causal convolution for context model.
    Only sees top-left context (raster order).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', torch.zeros_like(self.weight))
        self.create_mask()

    def create_mask(self):
        k = self.kernel_size[0]
        # Strict causal mask validation
        self.mask[:, :, :k//2, :] = 1
        self.mask[:, :, k//2, :k//2] = 1

    def forward(self, x):
        return F.conv2d(x, self.weight * self.mask, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class Hyperprior(nn.Module):
    """
    Elite Hyperprior Engine.
    Implements anti-aliased analysis, depthwise-separable parameters, 
    and stabilized GMM prediction.
    """
    def __init__(self, latent_dim=192, hyper_dim=128, num_components=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.hyper_dim = hyper_dim
        self.num_components = num_components
        
        # Hyper-analysis: y -> z
        self.ha_net = nn.Sequential(
            nn.Conv2d(latent_dim, hyper_dim, 3, stride=1, padding=1),
            nn.GroupNorm(8, hyper_dim),
            nn.GELU(),
            nn.Conv2d(hyper_dim, hyper_dim, 5, stride=2, padding=2),
            nn.GroupNorm(8, hyper_dim),
            nn.GELU(),
            nn.Conv2d(hyper_dim, hyper_dim, 5, stride=2, padding=2)
        )
        
        self.z_quantizer = SovereignQuantizer(hyper_dim)
        self.qvs = QVSUnitaryCoupling(hyper_dim)
        
        # Context model (causal)
        self.context_conv = MaskedConv2d(latent_dim, hyper_dim*2, 
                                         kernel_size=3, padding=1)
        
        # Hyper-synthesis: z_hat -> features
        # FIX 9: Optimized with Upsample+Conv to prevent probability artifacts
        self.hs_net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hyper_dim, hyper_dim, 3, padding=1),
            nn.GroupNorm(8, hyper_dim),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hyper_dim, hyper_dim, 3, padding=1),
            nn.GroupNorm(8, hyper_dim),
            nn.GELU(),
        )
        
        # Parameter prediction (FIX 7: Depthwise separable for 1.7K channel efficiency)
        self.param_net = nn.Sequential(
            nn.Conv2d(hyper_dim*3, hyper_dim*2, 1), # hyper_dim*3 = 384
            nn.GroupNorm(8, hyper_dim*2),
            nn.GELU(),
            nn.Conv2d(hyper_dim*2, hyper_dim*2, 3, padding=1, groups=hyper_dim*2),
            nn.GELU(),
            nn.Conv2d(hyper_dim*2, latent_dim * num_components * 3, 1)
        )
        
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
    
    def forward(self, y, hard_prob=None):
        """
        y: (B, latent_dim, H/16, W/16) — receives y_hat for honesty.
        """
        if hard_prob is None:
            hard_prob = not self.training
        # Hyper-analysis
        z = self.ha_net(y)
        
        # QVS Unitary Coupling on continuous space for better gradients
        z_mod_continuous = self.qvs(z)
        
        # Quantize hyper-latent
        z_hat, z_step = self.z_quantizer(z_mod_continuous, hard_prob=hard_prob)
        
        # Hyper-synthesis
        hs_features = self.hs_net(z_hat)
        
        # Residual skip for stability
        if hs_features.shape == z_hat.shape:
            hs_features = hs_features + z_hat
        
        return z_hat, z_step, hs_features

    def get_gmm_params(self, hs_features, ctx_features):
        # Raster-scan causality is enforced by MaskedConv2d.
        combined = torch.cat([hs_features, ctx_features], dim=1)
        params = self.param_net(combined)
        B, _, H, W = params.shape
        params = params.view(B, self.latent_dim, self.num_components, 3, H, W)
        
        weights = params[:, :, :, 0]
        means = params[:, :, :, 1]
        
        # Softplus for scale activation (stability)
        scales = F.softplus(params[:, :, :, 2]) * 0.1 + 1e-6
        
        # Soft GMM temperature (1.0) for stability
        temperature = 1.0
        weights = torch.softmax(weights / temperature, dim=2)
        
        return weights, means, scales
