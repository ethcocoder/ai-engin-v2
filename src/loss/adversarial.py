import torch
import torch.nn as nn
import torch.nn.functional as F

class AdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Use BCE with logits
        self.criterion = nn.BCEWithLogitsLoss()

    def discriminator_loss(self, real_preds, fake_preds):
        """
        L_adv = E[log D(x)] + E[log(1 - D(x_hat))]
        """
        loss = 0.0
        # Handle multi-scale discriminator outputs
        if not isinstance(real_preds, list):
            real_preds = [real_preds]
            fake_preds = [fake_preds]
            
        for real_pred, fake_pred in zip(real_preds, fake_preds):
            real_loss = self.criterion(real_pred, torch.ones_like(real_pred))
            fake_loss = self.criterion(fake_pred, torch.zeros_like(fake_pred))
            loss += (real_loss + fake_loss) * 0.5
            
        return loss

    def generator_loss(self, fake_preds):
        """
        L_G = -E[log D(x_hat)]  (implemented as BCE(D(x_hat), 1))
        """
        loss = 0.0
        if not isinstance(fake_preds, list):
            fake_preds = [fake_preds]
            
        for fake_pred in zip(fake_preds):
            # fake_pred is a tuple from zip, extract the element
            pred = fake_pred[0]
            loss += self.criterion(pred, torch.ones_like(pred))
            
        return loss
