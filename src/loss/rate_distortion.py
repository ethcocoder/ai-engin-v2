import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .perceptual import LPIPSLoss
from .entanglement import EntanglementRegularizer, SparsityRegularizer
from ..utils.metrics import ms_ssim

class RateDistortionLoss(nn.Module):
    """
    Elite Rate-Distortion Loss Engine.
    L = lambda * (L1 + MS-SSIM + LPIPS) + Rate + lambda * (Quantum + Sparsity)
    
    Professionally balanced for stable training and high perceptual fidelity.
    """
    def __init__(self, lmbda=0.01, use_ms_ssim=False, use_lpips=False, 
                 use_entanglement=False, entanglement_weight=0.01,
                 lpips_weight=0.1, lpips_warmup_epochs=10,
                 max_bpp=2.0, total_epochs=40, rate_warmup_pct=0.3):
        super().__init__()
        self.lmbda = lmbda
        self.use_ms_ssim = use_ms_ssim
        self.use_lpips = use_lpips
        self.use_entanglement = use_entanglement
        self.lpips_weight = lpips_weight
        self.lpips_warmup_epochs = lpips_warmup_epochs
        self.max_bpp = max_bpp
        self.current_epoch = 0
        self.total_epochs = total_epochs
        self.rate_warmup_pct = rate_warmup_pct
        
        self.l1_loss = nn.L1Loss()
        
        # Expert Regularizers
        self.entanglement_reg = EntanglementRegularizer(weight=entanglement_weight, use_renyi2=True)
        self.sparsity_reg = SparsityRegularizer(weight=entanglement_weight)
        
        if self.use_lpips:
            # We use the official or elite custom implementation
            self.lpips_loss = LPIPSLoss()
            
    def set_epoch(self, epoch):
        """Allows for loss schedules and warmups."""
        self.current_epoch = epoch
        
    def _normalize_for_ssim(self, x):
        """
        Force-normalizes images from the dataset's [-1, 1] range to [0, 1].
        This ensures MS-SSIM always operates on a standardized structural range.
        """
        return (x + 1) / 2
            
    def forward(self, x, x_hat, likelihoods, y=None):
        """
        Calculates the multi-component RD objective.
        """
        N, _, H, W = x.shape
        num_pixels = N * H * W
        
        # FIX 7: Rate warmup — let the encoder/decoder learn to reconstruct
        # before the entropy model starts pulling the latent distribution.
        # Scale rate weight from 0.01 → 1.0 over the first 30% of training.
        warmup_end = max(1, int(self.total_epochs * self.rate_warmup_pct))
        if self.current_epoch < warmup_end:
            rate_weight = 0.01 + 0.99 * (self.current_epoch / warmup_end)
        else:
            rate_weight = 1.0
        
        # 1. Rate (Bits Per Pixel)
        bpp_loss = 0.0
        for likelihood in likelihoods.values():
            # ELITE FIX: Numerically stable rate computation (-log2(p) = -ln(p)/ln(2))
            # No explicit clamping; epsilon inside log for stability.
            bpp = -torch.log(likelihood + 1e-10).sum() / (num_pixels * math.log(2))
            # FIX 5: Cap max bpp per component for stability
            bpp = torch.clamp(bpp, max=self.max_bpp)
            bpp_loss += bpp
            
        # 2. Distortion
        l1_val = self.l1_loss(x, x_hat)
        d_loss = l1_val
        
        # MS-SSIM (FIX 2: Range awareness)
        ms_ssim_val = None
        if self.use_ms_ssim:
            x_norm = self._normalize_for_ssim(x)
            x_hat_norm = self._normalize_for_ssim(x_hat)
            ms_ssim_val = ms_ssim(x_norm, x_hat_norm, data_range=1.0)
            d_loss += 0.5 * (1.0 - ms_ssim_val)
            
        # LPIPS (FIX 3: Warmup schedule)
        lpips_val = None
        if self.use_lpips and self.current_epoch >= self.lpips_warmup_epochs:
            lpips_val = self.lpips_loss(x, x_hat)
            # Linear weight annealing
            effective_weight = min(
                self.lpips_weight,
                0.05 + 0.05 * max(0, self.current_epoch - self.lpips_warmup_epochs) / 20
            )
            d_loss += effective_weight * lpips_val
            
        # 3. Regularization (Quantum & Sparsity)
        reg_loss = 0.0
        ent_val = 0.0
        spa_val = 0.0
        if y is not None and self.use_entanglement:
            ent_val = self.entanglement_reg(y)
            spa_val = self.sparsity_reg(y)
            reg_loss = ent_val + spa_val
            
        # FIX 4: Scale distortion and regularization by lambda to maintain priority balance
        # FIX 7: Apply rate_weight to prevent bpp from dominating early training
        total_loss = self.lmbda * d_loss + rate_weight * bpp_loss + self.lmbda * reg_loss
        
        # FIX 6: Comprehensive Metrics Return
        return {
            'loss': total_loss,
            'bpp_loss': bpp_loss.item() if isinstance(bpp_loss, torch.Tensor) else bpp_loss,
            'd_loss': d_loss.item() if isinstance(d_loss, torch.Tensor) else d_loss,
            'l1_loss': l1_val.item(),
            'ms_ssim_loss': (1.0 - ms_ssim_val).item() if ms_ssim_val is not None else 0.0,
            'lpips_loss': lpips_val.item() if lpips_val is not None else 0.0,
            'reg_loss': reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss,
            'ent_loss': ent_val.item() if isinstance(ent_val, torch.Tensor) else 0.0,
            'spa_loss': spa_val.item() if isinstance(spa_val, torch.Tensor) else 0.0
        }
