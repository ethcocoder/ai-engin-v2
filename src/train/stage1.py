import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import GradScaler, autocast
import torch.nn as nn
from tqdm import tqdm

from src.model.aether_codec import AetherCodec
from src.loss.rate_distortion import RateDistortionLoss
from src.utils.ema import EMA

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
    scaler = GradScaler('cuda')
    ema = EMA(model, decay=0.999)
    
    print("Starting Stage 1 Training...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, data in enumerate(pbar):
            # Assuming data is a tuple where first element is image
            x = data[0].to(device) if isinstance(data, (list, tuple)) else data.to(device)
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
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
            
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "bpp": f"{loss_dict['bpp_loss'].item():.4f}",
                    "mse": f"{loss_dict['d_loss'].item():.4f}"
                })
                
        scheduler.step()
        print(f"Epoch {epoch+1} Completed. Avg Loss: {epoch_loss/len(dataloader):.4f}")
        
    return model, ema

if __name__ == "__main__":
    import argparse
    from src.model.aether_codec import AetherCodec
    from src.train.dataset import get_dataloader
    import torch
    
    parser = argparse.ArgumentParser(description="Train AetherCodec Stage 1")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--data_dir", type=str, default="auto", help="Path to dataset (or 'auto' to download)")
    args = parser.parse_args()
    
    print(f"Initializing AetherCodec Stage 1 Training...")
    print(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}, Data: {args.data_dir}")
    
    model = AetherCodec()
    loader = get_dataloader(args.data_dir, batch_size=args.batch_size)
    model, ema = train_stage1(model, loader, epochs=args.epochs)
    
    torch.save(model.state_dict(), 'stage1_foundation.pth')
    print("Stage 1 complete and saved to 'stage1_foundation.pth'")
