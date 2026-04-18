"""
train_tpu.py — Paradox Genesis TPU-Accelerated Elite Training
=============================================================
Optimized for Google Colab TPU v2/v3 using torch_xla.
Enables High-Res Retina-Grade Manifold Learning.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

# --- TPU Specific Imports ---
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    
    # --- Paradox PJRT Force Logic ---
    if 'PJRT_DEVICE' not in os.environ:
        os.environ['PJRT_DEVICE'] = 'TPU'
        
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False
    print("[!] TPU libraries (torch_xla) not found. This script requires a TPU environment.")

from model import LatentGenesisCore
from data import get_dataloaders
from train import PerceptualLoss, ssim_loss

# --- Elite HD Training Logic ---
def train_loop(index, flags):
    """Distributed training loop for each TPU core."""
    torch.manual_seed(flags['seed'])
    
    # 1. Device Acquisition (TPU Core)
    device = xm.xla_device()
    print(f"[CORE {index}] Initiating Neural Synthesis...")

    # 2. Parallel Data Pipeline
    trainloader, testloader = get_dataloaders(
        batch_size=flags['batch_size'], 
        use_hd=True, 
        sample_limit=flags['sample_limit']
    )
    
    # Wrap with ParallelLoader for TPU efficiency
    train_device_loader = pl.ParallelLoader(trainloader, [device]).per_device_loader(device)

    # 3. Model & Optimizer
    model = LatentGenesisCore(latent_channels=flags['latent_channels']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=flags['lr'] * xm.xrt_world_size()) # Scale LR for TPU
    
    # 4. Losses
    perc_engine = PerceptualLoss().to(device).eval()
    
    # 5. Training Epochs
    for epoch in range(flags['epochs']):
        model.train()
        total_loss = 0
        
        for i, (images, _) in enumerate(train_device_loader):
            optimizer.zero_grad()
            
            # Forward Pass
            recon, mu, logvar = model(images)
            
            # Multi-Component Elite Loss
            l1_l = F.l1_loss(recon, images)
            s_l  = ssim_loss(recon, images)
            p_l  = perc_engine(recon, images)
            
            # KLD Stability Clamp
            logvar_c = torch.clamp(logvar, -10, 10)
            kld_l = -0.5 * torch.mean(1 + logvar_c - mu.pow(2) - logvar_c.exp())
            
            loss = l1_l + (0.5 * s_l) + (0.1 * p_l) + (0.001 * kld_l)
            
            # Targeted TPU Backprop
            loss.backward()
            xm.optimizer_step(optimizer) # Specialized TPU optimizer call
            
            if i % 20 == 0:
                xm.master_print(f"Epoch [{epoch+1}/{flags['epochs']}] | Batch {i} | Loss: {loss.item():.4f}")

        # Save Master Weights (Checkpointing)
        if index == 0: # Only the master core saves
            xm.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, "checkpoints/universal_tpu_master.pth")
            xm.master_print(f"[*] Master Weights Synchronized: Epoch {epoch+1}")

def main():
    if not TPU_AVAILABLE:
        return
    
    # Hyper-parameters for the TPU Rocket
    flags = {
        'batch_size': 32, # TPU can handle large batches
        'epochs': 100,
        'lr': 2e-4,
        'latent_channels': 16,
        'sample_limit': 10000,
        'seed': 42
    }
    
    # Spawn parallel processes (None = Auto-detect all TPU cores)
    print("[SYSTEM] Spawning Paradox Manifold Cores...")
    xmp.spawn(train_loop, args=(flags,), nprocs=None, start_method='fork')

if __name__ == "__main__":
    main()
