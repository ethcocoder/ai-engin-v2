import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast

from ..loss.rate_distortion import RateDistortionLoss
from ..loss.adversarial import AdversarialLoss
from ..model.discriminator import MultiScaleDiscriminator
from ..utils.ema import EMA

def train_stage3(model, dataloader, epochs=50, device='cuda', ema=None):
    """
    Stage 3: Add discriminator. Train full system: R + lambda*D + 0.1*L_G. lambda=0.1.
    Uses MS-SSIM + LPIPS for distortion.
    """
    model.to(device)
    # Unfreeze all
    for param in model.parameters():
        param.requires_grad = True
    model.train()
    
    discriminator = MultiScaleDiscriminator().to(device)
    discriminator.train()
    
    criterion = RateDistortionLoss(lmbda=0.1, use_ms_ssim=True, use_lpips=True).to(device)
    adv_criterion = AdversarialLoss().to(device)
    
    opt_G = AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)
    opt_D = AdamW(discriminator.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)
    
    scheduler_G = CosineAnnealingWarmRestarts(opt_G, T_0=50, T_mult=2)
    scheduler_D = CosineAnnealingWarmRestarts(opt_D, T_0=50, T_mult=2)
    
    scaler_G = GradScaler()
    scaler_D = GradScaler()
    
    if ema is None:
        ema = EMA(model, decay=0.999)
        
    print("Starting Stage 3 Training...")
    
    for epoch in range(epochs):
        for batch_idx, data in enumerate(dataloader):
            x = data[0].to(device) if isinstance(data, (list, tuple)) else data.to(device)
            
            # --- Train Discriminator ---
            opt_D.zero_grad()
            with autocast():
                with torch.no_grad():
                    x_hat, _, _ = model(x, force_hard=False)
                
                real_preds = discriminator(x)
                fake_preds = discriminator(x_hat.detach())
                
                d_loss = adv_criterion.discriminator_loss(real_preds, fake_preds)
                
            scaler_D.scale(d_loss).backward()
            scaler_D.unscale_(opt_D)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            scaler_D.step(opt_D)
            scaler_D.update()
            
            # --- Train Generator (Codec) ---
            opt_G.zero_grad()
            with autocast():
                x_hat, likelihoods, y = model(x, force_hard=False)
                fake_preds = discriminator(x_hat)
                
                rd_loss_dict = criterion(x, x_hat, likelihoods)
                g_adv_loss = adv_criterion.generator_loss(fake_preds)
                
                # Total loss: R + lambda*D + 0.1 * L_G
                total_g_loss = rd_loss_dict['loss'] + 0.1 * g_adv_loss
                
            scaler_G.scale(total_g_loss).backward()
            scaler_G.unscale_(opt_G)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_G.step(opt_G)
            scaler_G.update()
            
            ema.update(model)
            
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch {batch_idx} "
                      f"G_Loss: {total_g_loss.item():.4f} D_Loss: {d_loss.item():.4f} "
                      f"BPP: {rd_loss_dict['bpp_loss'].item():.4f}")
                
        scheduler_G.step()
        scheduler_D.step()
        print(f"Epoch {epoch+1} Completed.")
        
    return model, ema

if __name__ == "__main__":
    from src.model.aether_codec import AetherCodec
    from src.train.dataset import get_dataloader
    import torch
    import os
    
    print("Initializing AetherCodec Stage 3 Training (GAN)...")
    model = AetherCodec()
    if os.path.exists('stage2_refined.pth'):
        model.load_state_dict(torch.load('stage2_refined.pth', weights_only=True))
        print("Loaded Stage 2 weights.")
    else:
        print("Warning: stage2_refined.pth not found, starting from scratch.")
        
    loader = get_dataloader('auto', batch_size=8)
    model, ema = train_stage3(model, loader, epochs=50)
    
    torch.save(model.state_dict(), 'stage3_elite_final.pth')
    print("Stage 3 complete and saved to 'stage3_elite_final.pth'")
