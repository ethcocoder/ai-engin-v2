import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import GradScaler, autocast
import torch.nn as nn
from tqdm import tqdm
import warnings

from src.loss.rate_distortion import RateDistortionLoss
from src.model.qvs_flow import invalidate_qvs_cache
from src.utils.ema import EMA

def train_stage2(model, dataloader, epochs=100, device='cuda', ema=None, lmbda=0.005):
    """
    Stage 2: Perceptual Refinement.
    Switches to MS-SSIM + LPIPS loss and aggressively penalizes BPP to reach < 0.5.
    Unfreezes the Residual Refinement Network (RRN) for detail recovery.
    """
    model.to(device)
    model.train()
    
    # Ensure all components (including hyperprior) are trainable
    for param in model.parameters():
        param.requires_grad = True
            
    # Stage 2 Loss: L1 + MS-SSIM + LPIPS. Lower lambda (0.005) forces BPP down.
    criterion = RateDistortionLoss(
        lmbda=lmbda, 
        use_ms_ssim=True, 
        use_lpips=True, 
        use_entanglement=True, 
        total_epochs=epochs, 
        lpips_warmup_epochs=0, # Start LPIPS immediately for Stage 2
        rate_warmup_pct=0.0,    # Rate pressure is already active
        max_bpp=4.0
    ).to(device)
    
    params_to_train = [p for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamW(params_to_train, lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-4, 
        steps_per_epoch=len(dataloader), 
        epochs=epochs,
        pct_start=0.1, # Faster ramp-up
        div_factor=10
    )
    scaler = GradScaler('cuda')
    
    if ema is None:
        ema = EMA(model, decay=0.999)
        
    print(f"🚀 Starting Stage 2 Training (Target BPP < 0.5) for {epochs} epochs...")
    
    for epoch in range(epochs):
        criterion.set_epoch(epoch)
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, data in enumerate(pbar):
            x = data[0].to(device) if isinstance(data, (list, tuple)) else data.to(device)
            optimizer.zero_grad()
            
            hard_prob = 1.0 # Maintain production-ready quantization
            
            with autocast('cuda'):
                x_hat, likelihoods, metrics = model(x, hard_prob=hard_prob)
                loss_dict = criterion(x, x_hat, likelihoods, y=metrics.get('y_clean'))
                loss = loss_dict['loss']
            
            if torch.isnan(loss):
                optimizer.zero_grad()
                continue
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params_to_train, max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scheduler.step()
            
            invalidate_qvs_cache(model)
            ema.update(model)
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "bpp": f"{loss_dict['bpp_loss']:.4f}",
                    "psnr_sim": f"{loss_dict['d_loss']:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
                })
                
        print(f"Epoch {epoch+1} Completed. Avg Loss: {epoch_loss/len(dataloader):.4f}")

        # Checkpoint Management
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'ema_state_dict': ema.state_dict() if ema else None,
        }
        torch.save(checkpoint, 'stage2_latest.pth')
        
        if (epoch + 1) % 5 == 0:
            torch.save(checkpoint, f'stage2_epoch_{epoch+1}.pth')
            
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
    parser.add_argument("--no_clic", action="store_true", help="Disable CLIC 2020 dataset")
    parser.add_argument("--no_massive", action="store_true", help="Disable COCO + Flickr massive corpus")
    parser.add_argument("--use_flickr2k", action="store_true", help="Enable Flickr2K (requires ~20GB space)")
    args = parser.parse_args()
    
    print(f"Initializing AetherCodec Stage 2 Training...")
    print(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}, Data: {args.data_dir}")
    print(f"Dataset Config: Massive={'Disabled' if args.no_massive else 'Enabled'}, CLIC={'Disabled' if args.no_clic else 'Enabled'}, Flickr2K={'Enabled' if args.use_flickr2k else 'Disabled'}")
    
    model = AetherCodec()
    # Search for latest weights from Stage 1
    found_weights = None
    for f in ['stage1_foundation.pth', 'stage1_latest.pth', 'stage1_epoch_50.pth', 'stage1_epoch_10.pth']:
        if os.path.exists(f):
            found_weights = f
            break
            
    if found_weights:
        state = torch.load(found_weights, weights_only=True)
        # Handle full checkpoints vs raw state_dicts
        weights = state['state_dict'] if 'state_dict' in state else state
        model.load_state_dict(weights, strict=False)
        print(f"✅ Loaded Stage 1 weights from {found_weights}")
    else:
        print("⚠️ Warning: No Stage 1 weights found, starting from scratch.")
        
    loader = get_dataloader(
        args.data_dir, 
        batch_size=args.batch_size, 
        use_clic=not args.no_clic,
        use_10k_plus=not args.no_massive
    )
    model, ema = train_stage2(model, loader, epochs=args.epochs)
    
    torch.save(model.state_dict(), 'stage2_refined.pth')
    print("Stage 2 complete and saved to 'stage2_refined.pth'")
