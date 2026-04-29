import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import GradScaler, autocast

import torch.nn as nn
from tqdm import tqdm

from src.loss.rate_distortion import RateDistortionLoss
from src.loss.adversarial import AdversarialLoss
from src.model.discriminator import MultiScaleDiscriminator
from src.model.qvs_flow import invalidate_qvs_cache
from src.utils.ema import EMA

def train_stage3(model, dataloader, epochs=50, device='cuda', ema=None):
    """
    Stage 3: Elite Adversarial Training.
    Utilizes the expert-audited AdversarialLoss for maximum visual fidelity.
    """
    model.to(device)
    # Full model unfreeze for final refinement
    for param in model.parameters():
        param.requires_grad = True
    model.train()
    
    # 3-Scale Discriminator hierarchy
    discriminator = MultiScaleDiscriminator(num_scales=3).to(device)
    discriminator.train()
    
    # Loss Engines
    criterion = RateDistortionLoss(lmbda=0.1, use_ms_ssim=True, use_lpips=True, use_entanglement=True).to(device)
    adv_criterion = AdversarialLoss(lambda_fm=10.0).to(device)
    
    opt_G = AdamW(model.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    opt_D = AdamW(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=1e-4)
    
    scheduler_G = torch.optim.lr_scheduler.OneCycleLR(
        opt_G, max_lr=1e-4, steps_per_epoch=len(dataloader), epochs=epochs
    )
    scheduler_D = torch.optim.lr_scheduler.OneCycleLR(
        opt_D, max_lr=1e-4, steps_per_epoch=len(dataloader), epochs=epochs
    )
    
    scaler_G = GradScaler('cuda')
    scaler_D = GradScaler('cuda')
    
    if ema is None:
        ema = EMA(model, decay=0.999)
        
    print(f"Starting Stage 3 Elite Training for {epochs} epochs (GAN Mode)...")
    
    for epoch in range(epochs):
        # FIX: Synchronize loss engine with current epoch for warmups
        criterion.set_epoch(epoch)
        
        # Safety: Freeze RRN for foundation stability in early Stage 3
        if epoch < 5:
            for p in model.decoder.rrn.parameters():
                p.requires_grad = False
        else:
            for p in model.decoder.rrn.parameters():
                p.requires_grad = True
                
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, data in enumerate(pbar):
            x = data[0].to(device) if isinstance(data, (list, tuple)) else data.to(device)
            
            # FIX 5: Quantization Curriculum (Hard by default in Stage 3)
            hard_prob = 1.0 
            
            # --- Generate Fake Images ---
            with autocast('cuda'):
                x_hat, likelihoods, metrics_base = model(x, hard_prob=hard_prob)
            
            # --- Train Discriminator ---
            opt_D.zero_grad()
            with autocast('cuda'):
                real_preds, real_features = discriminator(x, return_features=True)
                fake_preds, fake_features = discriminator(x_hat.detach(), return_features=True)
                
                # Using expert-audited D loss (with label smoothing)
                d_loss = adv_criterion.discriminator_loss(real_preds, fake_preds)
            
            scaler_D.scale(d_loss).backward()
            scaler_D.unscale_(opt_D)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=0.5)
            scaler_D.step(opt_D)
            scaler_D.update()
            
            # --- Train Generator (Codec) ---
            opt_G.zero_grad()
            with autocast('cuda'):
                # Fresh predictions for G gradient flow
                fake_preds, fake_features = discriminator(x_hat, return_features=True)
                # Need real features for FM loss
                with torch.no_grad():
                    _, real_features = discriminator(x, return_features=True)
                
                # Using expert-audited G loss (Adv + FM)
                _, g_adv_total, adv_metrics = adv_criterion(real_preds, fake_preds, real_features, fake_features)
                
                # Core Rate-Distortion Loss
                rd_loss_dict = criterion(x, x_hat, likelihoods, y=metrics_base.get('y_clean'))
                
                # Final loss combination
                total_g_loss = rd_loss_dict['loss'] + g_adv_total * 0.1 # Reduced Adv weight for stability

            scaler_G.scale(total_g_loss).backward()
            scaler_G.unscale_(opt_G)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler_G.step(opt_G)
            scaler_G.update()
            
            # Schedulers step per batch
            scheduler_G.step()
            scheduler_D.step()
            
            # FIX 1: Invalidate QVS cache after updates
            invalidate_qvs_cache(model)
            
            ema.update(model)
            
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    "G": f"{total_g_loss.item():.3f}",
                    "D": f"{d_loss.item():.3f}",
                    "bpp": f"{rd_loss_dict['bpp_loss']:.3f}",
                    "lr": f"{opt_G.param_groups[0]['lr']:.2e}"
                })
        
        # Save Checkpoints
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'ema_state_dict': ema.state_dict() if ema else None,
            'discriminator_state_dict': discriminator.state_dict(),
        }
        torch.save(checkpoint, 'stage3_latest.pth')
        if (epoch + 1) % 5 == 0:
            torch.save(checkpoint, f'stage3_epoch_{epoch+1}.pth')
        
    return model, ema

if __name__ == "__main__":
    import argparse
    from src.model.aether_codec import AetherCodec
    from src.train.dataset import get_dataloader
    
    parser = argparse.ArgumentParser(description="Train AetherCodec Stage 3")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--data_dir", type=str, default="auto", help="Path to dataset")
    args = parser.parse_args()
    
    model = AetherCodec()
    if os.path.exists('stage2_refined.pth'):
        model.load_state_dict(torch.load('stage2_refined.pth', weights_only=True))
        print("Loaded Stage 2 weights.")
        
    loader = get_dataloader(args.data_dir, batch_size=args.batch_size)
    model, ema = train_stage3(model, loader, epochs=args.epochs)
    
    torch.save(model.state_dict(), 'stage3_elite_final.pth')
    print("Stage 3 complete.")
