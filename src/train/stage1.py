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

def train_stage1(model, dataloader, epochs=100, device='cuda', lmbda=0.01):
    """
    Stage 1: Foundation Training — Encoder + Decoder + Hyperprior.
    
    CRITICAL FIX v6: Proper compression training.
    - lambda=0.01 for aggressive compression (small files)
    - lambda=0.05 for balanced quality/size
    - Rate warmup is SHORT (10%) to start compressing immediately
    - Hard quantization curriculum ramps faster
    """
    model.to(device)
    model.train()
    
    # FIX 8: Freeze RRN for Stage 1 (let foundation learn first)
    for p in model.decoder.rrn.parameters():
        p.requires_grad = False
    
    # CRITICAL FIX v6: rate_warmup_pct=0.1 (was 0.3) for fast compression
    criterion = RateDistortionLoss(
        lmbda=lmbda,
        use_ms_ssim=False, 
        use_lpips=False, 
        use_entanglement=True, 
        total_epochs=epochs,
        rate_warmup_pct=0.1  # Only 10% warmup — rate kicks in fast
    ).to(device)
    
    # ELITE ACCURACY: OneCycleLR for faster convergence and better optima
    optimizer = AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-4, 
        steps_per_epoch=len(dataloader), 
        epochs=epochs,
        pct_start=0.3, # 30% of time warming up
        div_factor=10, # Start at 1e-5
        final_div_factor=100 # End at 1e-6
    )
    scaler = GradScaler('cuda')
    ema = EMA(model, decay=0.999)
        
    print(f"Starting Stage 1 Training for {epochs} epochs (lambda={lmbda})...")
    print(f"  Lambda={lmbda}: {'Aggressive compression' if lmbda <= 0.01 else 'Balanced' if lmbda <= 0.05 else 'High quality'}")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_bpp = 0.0
        epoch_distortion = 0.0
        
        # FIX: Synchronize loss engine with current epoch for warmups
        criterion.set_epoch(epoch)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, data in enumerate(pbar):
            # Assuming data is a tuple where first element is image
            x = data[0].to(device) if isinstance(data, (list, tuple)) else data.to(device)
            
            optimizer.zero_grad()
            
            # CRITICAL FIX v6: Faster quantization curriculum
            # Start hard quantization earlier so the model learns to work with
            # actual quantized values, not just noise-perturbed ones
            # Ramp from 0→1 over epochs 20%-70% of training
            hard_prob = min(1.0, max(0.0, (epoch - epochs*0.2) / (epochs*0.5 + 1e-6)))
            
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
            
            # Gradient clipping (Norm + Value for Lockdown Build)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # OneCycle scheduler step per batch
            scheduler.step()
            
            # FIX 1: Invalidate QVS cache after weight update
            invalidate_qvs_cache(model)
            
            ema.update(model)
            
            epoch_loss += loss.item()
            epoch_bpp += loss_dict['bpp_loss']
            epoch_distortion += loss_dict['d_loss']
            
            if batch_idx % 10 == 0:
                # Show rate_weight so user can verify warmup progress
                warmup_end = max(1, int(epochs * 0.1))
                rw = min(1.0, 0.1 + 0.9 * (epoch / warmup_end)) if epoch < warmup_end else 1.0
                
                # Show step_size stats for compression monitoring
                y_step_mean = model.y_quantizer.step_size.data.mean().item()
                
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "bpp": f"{loss_dict['bpp_loss']:.3f}",
                    "dist": f"{loss_dict['d_loss']:.4f}",
                    "step": f"{y_step_mean:.2f}",
                    "rw": f"{rw:.2f}",
                    "hq": f"{hard_prob:.2f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
                })
                
        avg_loss = epoch_loss / len(dataloader)
        avg_bpp = epoch_bpp / len(dataloader)
        avg_dist = epoch_distortion / len(dataloader)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | BPP: {avg_bpp:.3f} | Distortion: {avg_dist:.4f}")
        
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
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (minimum 50 recommended)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lmbda", type=float, default=0.01, help="Lambda: 0.001=extreme compression, 0.01=aggressive, 0.05=balanced, 0.1=high quality")
    parser.add_argument("--data_dir", type=str, default="auto", help="Path to dataset (or 'auto' to download)")
    parser.add_argument("--no_clic", action="store_true", help="Disable CLIC 2020 dataset")
    parser.add_argument("--no_massive", action="store_true", help="Disable COCO + Flickr massive corpus")
    parser.add_argument("--use_flickr2k", action="store_true", help="Enable Flickr2K (requires ~20GB space)")
    args = parser.parse_args()
    
    print(f"Initializing AetherCodec Stage 1 Training...")
    print(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}, Lambda: {args.lmbda}")
    print(f"Dataset Config: Massive={'Disabled' if args.no_massive else 'Enabled'}, CLIC={'Disabled' if args.no_clic else 'Enabled'}, Flickr2K={'Enabled' if args.use_flickr2k else 'Disabled'}")
    
    model = AetherCodec()
    # If the user has a previous checkpoint, they might want to resume
    if os.path.exists('stage1_latest.pth'):
        print("Found existing checkpoint. Resuming...")
        checkpoint = torch.load('stage1_latest.pth', map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])

    loader = get_dataloader(
        args.data_dir, 
        batch_size=args.batch_size, 
        use_clic=not args.no_clic,
        use_10k_plus=not args.no_massive
    )
    
    model, ema = train_stage1(model, loader, epochs=args.epochs, lmbda=args.lmbda)
    
    torch.save(model.state_dict(), 'stage1_foundation.pth')
    print("Stage 1 complete and saved to 'stage1_foundation.pth'")
