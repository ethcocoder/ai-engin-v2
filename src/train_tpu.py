"""
train_tpu.py — Paradox Genesis v5e Direct-Attached Engine
=========================================================
Optimized for Single-Node TPU v5e-1 (Colab PJRT Runtime).
Bypasses the cluster address mismatch by using Direct Synthesis.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- TPU v5e Direct Activation ---
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    
    # Force PJRT to ignore clusters and focus on local hardware
    os.environ['PJRT_DEVICE'] = 'TPU'
    os.environ['XRT_TPU_CONFIG'] = "localservice;0;localhost:51011"
    
    # --- Paradox Speed Boost (2x Multiply) ---
    os.environ['XLA_USE_BF16'] = '1'
    
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False

from model import LatentGenesisCore
from data import get_dataloaders
from train import PerceptualLoss, ssim_loss

def train_tpu_direct(flags):
    """Direct-attached training loop for v5e-1."""
    device = torch_xla.device()
    print(f"[*] Paradox Sovereign v5e Engine Activated on {device}", flush=True)

    # 1. Pipeline Acquisition
    trainloader, _ = get_dataloaders(
        batch_size=flags['batch_size'], 
        use_hd=True, 
        sample_limit=flags['sample_limit']
    )
    
    # 2. Neural Manifold Initialization
    model = LatentGenesisCore(latent_channels=flags['latent_channels']).to(device)
    # ELITE UPGRADE: 5x Speed boost. The 20-epoch cycle requires a far more aggressive initial descent.
    # Betas tuned for rapid momentum decay.
    optimizer = optim.Adam(model.parameters(), lr=flags['lr'] * 5.0, betas=(0.5, 0.999), eps=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=flags['epochs'])
    perc_engine = PerceptualLoss().to(device).eval()
    
    # 3. Training Loop
    for epoch in range(flags['epochs']):
        model.train()
        
        # --- Paradox Stream Refresh ---
        train_device_loader = pl.ParallelLoader(trainloader, [device]).per_device_loader(device)
        
        # Professional Progress Bar
        pbar = tqdm(total=len(trainloader), desc=f"Epoch {epoch+1}/{flags['epochs']}", unit="batch")
        
        for i, (images, _) in enumerate(train_device_loader):
            optimizer.zero_grad()
            
            recon, mu, logvar = model(images)
            
            # ELITE LOSS SYNTHESIS:
            # Gradients MUST flow. Removing clamps allows the network to physically fix major errors 
            # instead of hitting '0-gradient' dead-zones in the math.
            l1_l = F.l1_loss(recon, images)
            s_l  = ssim_loss(recon, images)
            p_l  = perc_engine(recon, images)
            
            # KLD (Already mathematically clamped in model.py to prevent overflow)
            kld_l = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            # Master Ratio 2.0: 
            # - Massive L1 (50.0) ensures blueprint boundaries are mathematically rigid.
            # - SSIM (1.0) forces precise structural geometry projection.
            loss = (l1_l * 50.0) + (s_l * 1.0) + (p_l * 0.01) + (kld_l * 0.0001)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5) # Tightened bound controls 5x LR
            xm.optimizer_step(optimizer)
            torch_xla.sync() # Elite TPU graph sync mathematically mandates pristine tensor resolution
            
            # Real-time Telemetry
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.6f}")
            pbar.update(1)

        pbar.close()
        scheduler.step()

        # Sync Master Weights
        save_path = "checkpoints/universal_tpu_master.pth"
        os.makedirs("checkpoints", exist_ok=True)
        xm.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }, save_path)
        print(f"--- [MASTER SYNCHRONIZED] Checkpoint saved: {save_path} ---", flush=True)

if __name__ == "__main__":
    if TPU_AVAILABLE:
        # TEST-FLIGHT CONFIGURATION
        flags = {
            'batch_size': 32,
            'epochs': 20, # Efficient 20-epoch cycle for 16KB speed
            'lr': 1e-4,
            'latent_channels': 64, # Upgraded to 64 (16KB) for Sovereign High-Fidelity transmission
            'sample_limit': 10000
        }
        train_tpu_direct(flags)
