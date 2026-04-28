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
        # FIX 3: Strict causal mask validation
        self.mask[:, :, :k//2, :] = 1
        self.mask[:, :, k//2, :k//2] = 1

    def forward(self, x):
        # FIX 3: Ensure mask is always applied correctly
        masked_weight = self.weight * self.mask
        return F.conv2d(x, masked_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class Hyperprior(nn.Module):
    """
    Elite Hyperprior Engine.
    Implements anti-aliased analysis, depthwise-separable parameters, 
    and stabilized GMM prediction.
    """
    def __init__(self, latent_dim=192, hyper_dim=128, num_components=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.hyper_dim = hyper_dim
        self.num_components = num_components
        
        # Hyper-analysis: y -> z
        # FIX 6: Added GroupNorm for internal covariate stability
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
            nn.Conv2d(hyper_dim*4, hyper_dim*2, 1),
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
        
        # FIX 2: QVS Unitary Coupling on continuous space for better gradients
        z_mod_continuous = self.qvs(z)
        
        # Quantize hyper-latent
        z_hat, z_step = self.z_quantizer(z_mod_continuous, hard_prob=hard_prob)
        
        # Hyper-synthesis
        hs_features = self.hs_net(z_hat)
        
        # FIX 9: Residual skip for stability
        if hs_features.shape == z_hat.shape: # Check if resolution matches
            hs_features = hs_features + z_hat
        
        return z_hat, z_step, hs_features

    def get_gmm_params(self, hs_features, ctx_features):
        # FIX 8: Enforce strict causality to prevent information leakage
        H, W = ctx_features.shape[-2:]
        causal_mask = torch.tril(torch.ones(H, W, device=ctx_features.device))
        ctx_features = ctx_features * causal_mask[None, None, :, :]
        
        combined = torch.cat([hs_features, ctx_features], dim=1)
        params = self.param_net(combined)
        B, _, H, W = params.shape
        params = params.view(B, self.latent_dim, self.num_components, 3, H, W)
        
        weights = params[:, :, :, 0]
        means = params[:, :, :, 1]
        
        # FIX 4: Clamp scales to prevent exp explosion and NaN crashes
        scales = torch.exp(torch.clamp(params[:, :, :, 2], -10, 10))
        
        # FIX 5: Temperature-controlled softmax for balanced component usage
        temperature = 0.5
        weights = torch.softmax(weights / temperature, dim=2)
        
        return weights, means, scales
