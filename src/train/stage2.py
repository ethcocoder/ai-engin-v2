import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import GradScaler, autocast
import torch.nn as nn
from tqdm import tqdm

from src.loss.rate_distortion import RateDistortionLoss
from src.utils.ema import EMA

def train_stage2(model, dataloader, epochs=100, device='cuda', ema=None):
    """
    Stage 2: Switch distortion to MS-SSIM. lambda=0.05. 
    Freeze hyperprior, train main codec.
    """
    model.to(device)
    model.train()
    
    # Freeze hyperprior
    if model.use_hyperprior:
        for param in model.hyperprior.parameters():
            param.requires_grad = False
            
    criterion = RateDistortionLoss(lmbda=0.01, use_ms_ssim=True, use_lpips=True, use_entanglement=True).to(device)
    
    # Train only main codec parameters (encoder, decoder, quantizer)
    params_to_train = [p for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamW(params_to_train, lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    scaler = GradScaler('cuda')
    
    if ema is None:
        ema = EMA(model, decay=0.999)
        
    print("Starting Stage 2 Training...")
    
    for epoch in range(epochs):
        # FIX 8: Freeze RRN for first 20 epochs
        if epoch < 20:
            for p in model.decoder.rrn.parameters():
                p.requires_grad = False
        else:
            for p in model.decoder.rrn.parameters():
                p.requires_grad = True
                
        # FIX: Synchronize loss engine with current epoch for warmups
        criterion.set_epoch(epoch)
        
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, data in enumerate(pbar):
            x = data[0].to(device) if isinstance(data, (list, tuple)) else data.to(device)
            
            optimizer.zero_grad()
            
            # FIX 5: Quantization Curriculum
            hard_prob = min(1.0, max(0.0, (epoch - epochs*0.3) / (epochs*0.4 + 1e-6)))
            
            with autocast('cuda'):
                x_hat, likelihoods, metrics = model(x, hard_prob=hard_prob)
                loss_dict = criterion(x, x_hat, likelihoods, y=metrics.get('y_clean'))
                loss = loss_dict['loss']
            
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params_to_train, max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # FIX 1: Invalidate QVS cache after weight update
            invalidate_qvs_cache(model)
            
            ema.update(model)
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "bpp": f"{loss_dict['bpp_loss'].item():.4f}",
                    "ms-ssim": f"{loss_dict['d_loss'].item():.4f}"
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
        torch.save(checkpoint, 'stage2_latest.pth')
        
        # Save "Milestone" every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, f'stage2_epoch_{epoch+1}.pth')
            print(f"Milestone checkpoint saved: stage2_epoch_{epoch+1}.pth")
        
    return model, ema

if __name__ == "__main__":
    import argparse
    from src.model.aether_codec import AetherCodec
    from src.train.dataset import get_dataloader
    import torch
    import os
    
    parser = argparse.ArgumentParser(description="Train AetherCodec Stage 2")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--data_dir", type=str, default="auto", help="Path to dataset")
    args = parser.parse_args()
    
    print(f"Initializing AetherCodec Stage 2 Training...")
    print(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}, Data: {args.data_dir}")
    
    model = AetherCodec()
    if os.path.exists('stage1_foundation.pth'):
        model.load_state_dict(torch.load('stage1_foundation.pth', weights_only=True))
        print("Loaded Stage 1 weights.")
    else:
        print("Warning: stage1_foundation.pth not found, starting from scratch.")
        
    loader = get_dataloader(args.data_dir, batch_size=args.batch_size)
    model, ema = train_stage2(model, loader, epochs=args.epochs)
    
    torch.save(model.state_dict(), 'stage2_refined.pth')
    print("Stage 2 complete and saved to 'stage2_refined.pth'")
