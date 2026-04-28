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
from src.model.qvs_flow import invalidate_qvs_cache
from src.loss.rate_distortion import RateDistortionLoss
from src.utils.ema import EMA

def train_stage1(model, dataloader, epochs=100, device='cuda'):
    """
    Stage 1: Train only encoder+decoder+hyperprior. 
    Loss: R + lambda*MSE. lambda=0.01.
    """
    model.to(device)
    model.train()
    
    # FIX 8: Freeze RRN for Stage 1 (let foundation learn first)
    for p in model.decoder.rrn.parameters():
        p.requires_grad = False
        
    criterion = RateDistortionLoss(lmbda=0.01, use_ms_ssim=False, use_lpips=False, use_entanglement=True).to(device)
    # FIX: Lower LR (5e-5) for 'Honest' architecture stability
    optimizer = AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    scaler = GradScaler('cuda')
    ema = EMA(model, decay=0.999)
        
    print("Starting Stage 1 Training...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        # FIX: Synchronize loss engine with current epoch for warmups
        criterion.set_epoch(epoch)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, data in enumerate(pbar):
            # Assuming data is a tuple where first element is image
            x = data[0].to(device) if isinstance(data, (list, tuple)) else data.to(device)
            
            optimizer.zero_grad()
            
            # FIX 5: Quantization Curriculum
            hard_prob = min(1.0, max(0.0, (epoch - epochs*0.5) / (epochs*0.3 + 1e-6)))
            
            with autocast('cuda'):
                x_hat, likelihoods, metrics = model(x, hard_prob=hard_prob)
                loss_dict = criterion(x, x_hat, likelihoods, y=metrics.get('y_clean'))
                loss = loss_dict['loss']
            
            # ELITE AUDIT v5: NaN Safety Guard
            if torch.isnan(loss):
                print(f"⚠️ Warning: NaN Loss detected in Batch {batch_idx}. Skipping...")
                optimizer.zero_grad()
                continue
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # FIX 1: Invalidate QVS cache after weight update
            invalidate_qvs_cache(model)
            
            ema.update(model)
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "bpp": f"{loss_dict['bpp_loss']:.4f}",
                    "mse": f"{loss_dict['d_loss']:.4f}"
                })
                
        scheduler.step()
        print(f"Epoch {epoch+1} Completed. Avg Loss: {epoch_loss/len(dataloader):.4f}")
        
        # --- Checkpoint Saving ---
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'ema_state_dict': ema.state_dict() if ema else None,
            'optimizer_state_dict': optimizer.state_dict(),
        }
        
        # Save "Latest" (Overwrites every epoch)
        torch.save(checkpoint, 'stage1_latest.pth')
        
        # Save "Milestone" every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, f'stage1_epoch_{epoch+1}.pth')
            print(f"Milestone checkpoint saved: stage1_epoch_{epoch+1}.pth")
        
    return model, ema

if __name__ == "__main__":
    import argparse
    from src.model.aether_codec import AetherCodec
    from src.train.dataset import get_dataloader
    import torch
    
    parser = argparse.ArgumentParser(description="Train AetherCodec Stage 1")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
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
