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

# --- TPU v5e Direct Activation ---
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    
    # Force PJRT to ignore clusters and focus on local hardware
    os.environ['PJRT_DEVICE'] = 'TPU'
    os.environ['XRT_TPU_CONFIG'] = "localservice;0;localhost:51011"
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False

from model import LatentGenesisCore
from data import get_dataloaders
from train import PerceptualLoss, ssim_loss

def train_tpu_direct(flags):
    """Direct-attached training loop for v5e-1."""
    device = xm.xla_device()
    print(f"[*] Paradox Sovereign v5e Engine Activated on {device}")

    # 1. Pipeline Acquisition
    trainloader, _ = get_dataloaders(
        batch_size=flags['batch_size'], 
        use_hd=True, 
        sample_limit=flags['sample_limit']
    )
    
    # Paradox Parallel Loader (The secret to v5e speed)
    train_device_loader = pl.ParallelLoader(trainloader, [device]).per_device_loader(device)

    # 2. Neural Manifold Initialization
    model = LatentGenesisCore(latent_channels=flags['latent_channels']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=flags['lr'])
    perc_engine = PerceptualLoss().to(device).eval()
    
    # 3. Phase 2 Synthesis
    for epoch in range(flags['epochs']):
        model.train()
        for i, (images, _) in enumerate(train_device_loader):
            optimizer.zero_grad()
            
            recon, mu, logvar = model(images)
            
            # Loss Synthesis (Phase 2.3 High-Stability Calibration)
            l1_l = F.l1_loss(recon, images)
            s_l  = ssim_loss(recon, images)
            p_l  = perc_engine(recon, images)
            
            # KLD Stability Gate (Crucial for preventing green collapse)
            logvar_c = torch.clamp(logvar, -10, 10)
            kld_l = -0.5 * torch.mean(1 + logvar_c - mu.pow(2) - logvar_c.exp())
            
            # Hybrid Elite Loss: 10x L1 Bedrock + Subtle Texture Polishing
            # This ratio prevents the "Green Noise" collapse 
            loss = (l1_l * 10.0) + (s_l * 0.05) + (p_l * 0.01) + (kld_l * 0.0001)
            
            # --- Paradox Stability Gate ---
            loss.backward()
            
            # Explicit Gradient Clipping to prevent "Checkerboard Chaos"
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            xm.optimizer_step(optimizer)
            
            if i == 0:
                print(f"[*] Neural Reality Anchored. Sync loop active.", flush=True)
            
            if i % 20 == 0:
                print(f"Epoch [{epoch+1}/{flags['epochs']}] | Batch {i} | Loss: {loss.item():.4f}", flush=True)

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
        flags = {
            'batch_size': 32,
            'epochs': 100,
            'lr': 1e-4,
            'latent_channels': 16,
            'sample_limit': 10000
        }
        train_tpu_direct(flags)
