import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn

from ..model.aether_codec import AetherCodec
from ..loss.rate_distortion import RateDistortionLoss
from ..utils.ema import EMA

def train_stage1(model, dataloader, epochs=100, device='cuda'):
    """
    Stage 1: Train only encoder+decoder+hyperprior. 
    Loss: R + lambda*MSE. lambda=0.01.
    """
    model.to(device)
    model.train()
    
    criterion = RateDistortionLoss(lmbda=0.01, use_ms_ssim=False, use_lpips=False).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    scaler = GradScaler()
    ema = EMA(model, decay=0.999)
    
    print("Starting Stage 1 Training...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, data in enumerate(dataloader):
            # Assuming data is a tuple where first element is image
            x = data[0].to(device) if isinstance(data, (list, tuple)) else data.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                x_hat, likelihoods, y = model(x, force_hard=False)
                loss_dict = criterion(x, x_hat, likelihoods)
                loss = loss_dict['loss']
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            ema.update(model)
            
            epoch_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch {batch_idx} Loss: {loss.item():.4f} "
                      f"BPP: {loss_dict['bpp_loss'].item():.4f} MSE: {loss_dict['d_loss'].item():.4f}")
                
        scheduler.step()
        print(f"Epoch {epoch+1} Completed. Avg Loss: {epoch_loss/len(dataloader):.4f}")
        
    return model, ema

if __name__ == "__main__":
    from src.model.aether_codec import AetherCodec
    from src.train.dataset import get_dataloader
    import torch
    
    print("Initializing AetherCodec Stage 1 Training...")
    model = AetherCodec()
    loader = get_dataloader('auto', batch_size=8)
    model, ema = train_stage1(model, loader, epochs=100)
    
    torch.save(model.state_dict(), 'stage1_foundation.pth')
    print("Stage 1 complete and saved to 'stage1_foundation.pth'")
