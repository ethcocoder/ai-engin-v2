import torch
import torch.nn as nn
import torch.nn.functional as F

class AdversarialLoss(nn.Module):
    """
    Elite Adversarial Loss Engine for Stage 3 Training.
    Implements Label Smoothing, Multi-Scale Feature Matching, and Scale Weighting.
    """
    def __init__(self, lambda_fm=10.0, label_smooth_real=0.9, label_smooth_fake=0.1):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.lambda_fm = lambda_fm
        self.smooth_real = label_smooth_real
        self.smooth_fake = label_smooth_fake
        self.l1 = nn.L1Loss()

    def feature_matching_loss(self, real_features, fake_features):
        """
        L_FM = sum_i (1/N_i) * ||D_i(real) - D_i(fake)||_1
        Stops the generator from 'cheating' by matching high-level discriminator traits.
        """
        if real_features is None or fake_features is None:
            return 0.0
        
        loss = 0.0
        # real_features is a list of lists: [scale1_feats, scale2_feats, ...]
        for rf_scale, ff_scale in zip(real_features, fake_features):
            for rf, ff in zip(rf_scale, ff_scale):
                # Normalized L1 loss
                loss += self.l1(rf, ff)
        
        return loss * self.lambda_fm

    def discriminator_loss(self, real_preds, fake_preds, weights=None):
        """
        Computes the loss for the Discriminator.
        Encourages D(real) -> smooth_real and D(fake) -> smooth_fake.
        """
        loss = 0.0
        
        if not isinstance(real_preds, list):
            real_preds = [real_preds]
            fake_preds = [fake_preds]
        
        # Safety detach to ensure D only learns from its own predictions
        fake_preds_detached = [fp.detach() for fp in fake_preds]
        
        # FIX 4: Default weights for multi-scale (H/1, H/2, H/4)
        if weights is None:
            weights = [1.0, 0.8, 0.5] 
        
        for i, (rp, fp) in enumerate(zip(real_preds, fake_preds_detached)):
            # FIX 1: Label Smoothing
            real_labels = torch.ones_like(rp) * self.smooth_real
            fake_labels = torch.zeros_like(fp) + self.smooth_fake
            
            real_loss = self.criterion(rp, real_labels)
            fake_loss = self.criterion(fp, fake_labels)
            
            # Weighted average across scales
            w = weights[i] if i < len(weights) else 1.0
            loss += w * (real_loss + fake_loss) * 0.5
        
        return loss

    def generator_loss(self, fake_preds, weights=None):
        """
        Computes the loss for the Generator.
        Encourages D(fake) -> 1.0 (fool the discriminator).
        """
        loss = 0.0
        
        if not isinstance(fake_preds, list):
            fake_preds = [fake_preds]
        
        if weights is None:
            weights = [1.0, 0.8, 0.5]
        
        for i, fp in enumerate(fake_preds):
            w = weights[i] if i < len(weights) else 1.0
            # Generator always wants the 'Hard' target (1.0) to beat the discriminator
            loss += w * self.criterion(fp, torch.ones_like(fp))
        
        return loss

    def forward(self, real_preds, fake_preds, real_features=None, fake_features=None):
        """
        Main entry point for Stage 3 loss calculation.
        Returns total losses and monitoring metrics.
        """
        d_loss = self.discriminator_loss(real_preds, fake_preds)
        g_adv_loss = self.generator_loss(fake_preds)
        
        # FIX 2: Feature Matching Loss
        fm_loss = self.feature_matching_loss(real_features, fake_features)
        g_loss_total = g_adv_loss + fm_loss
        
        # FIX 8: Metrics Dashboard
        metrics = {
            'd_loss': d_loss.item(),
            'g_adv': g_adv_loss.item(),
            'fm_loss': fm_loss.item() if isinstance(fm_loss, torch.Tensor) else fm_loss,
        }
        
        # Monitor confidence (how sure is D about its guess?)
        with torch.no_grad():
            # Check the finest scale (first scale)
            if isinstance(real_preds, list):
                rp = real_preds[0]
                fp = fake_preds[0]
            else:
                rp = real_preds
                fp = fake_preds
            
            metrics['d_real_conf'] = torch.sigmoid(rp).mean().item()
            metrics['d_fake_conf'] = torch.sigmoid(fp).mean().item()
        
        return d_loss, g_loss_total, metrics
