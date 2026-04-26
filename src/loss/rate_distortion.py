import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .perceptual import LPIPSLoss
from ..utils.metrics import ms_ssim

class RateDistortionLoss(nn.Module):
    """
    Rate-Distortion loss for training the codec.
    L = lambda_bpp * R + D
    where R is the estimated entropy (bits per pixel)
    and D is the multi-scale distortion term.
    """
    def __init__(self, lmbda=0.01, use_ms_ssim=False, use_lpips=False):
        super().__init__()
        self.lmbda = lmbda
        self.use_ms_ssim = use_ms_ssim
        self.use_lpips = use_lpips
        
        self.l1_loss = nn.L1Loss()
        
        if self.use_lpips:
            self.lpips_loss = LPIPSLoss()
            
    def forward(self, x, x_hat, likelihoods):
        """
        x: original image (B, 3, H, W)
        x_hat: reconstructed image (B, 3, H, W)
        likelihoods: dictionary with 'y' and optionally 'z' likelihoods
        """
        N, _, H, W = x.shape
        num_pixels = N * H * W
        
        # 1. Rate (Bits Per Pixel)
        # -log2(p) to get bits
        bpp_loss = 0.0
        for likelihood in likelihoods.values():
            # likelihoods are probabilities, we convert to bits
            bpp = torch.log(likelihood).sum() / (-math.log(2) * num_pixels)
            bpp_loss += bpp
            
        # 2. Distortion
        # L1 Loss
        d_loss = self.l1_loss(x, x_hat)
        
        # MS-SSIM Loss
        if self.use_ms_ssim:
            ms_ssim_val = ms_ssim(x, x_hat, data_range=1.0)
            d_loss += 0.5 * (1.0 - ms_ssim_val)
            
        # LPIPS Loss
        if self.use_lpips:
            lpips_val = self.lpips_loss(x, x_hat)
            d_loss += 0.1 * lpips_val
            
        # Total Loss
        total_loss = self.lmbda * d_loss + bpp_loss
        
        return {
            'loss': total_loss,
            'bpp_loss': bpp_loss,
            'd_loss': d_loss
        }
