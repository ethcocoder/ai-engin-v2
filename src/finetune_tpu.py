"""
finetune_tpu.py — Paradox Retina-Grade Fine-tuning Engine
=========================================================
Loads the structurally-perfect TPU Master checkpoint and fine-tunes
it EXCLUSIVELY on native high-resolution imagery to learn high-frequency
textures and eliminate blur.
"""

import sys
import os
import argparse
import urllib.request
import tarfile
from pathlib import Path

# --- Advanced Pathing Protocol ---
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from model import LatentGenesisCore
from train import PerceptualLoss, ssim_loss

# --- TPU v5e Direct Activation ---
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    os.environ['PJRT_DEVICE'] = 'TPU'
    os.environ['XRT_TPU_CONFIG'] = "localservice;0;localhost:51011"
    # Paradox Stability Guard: Disable Auto-BF16 to prevent Mixed-Precision crashes 
    os.environ['XLA_USE_BF16'] = '0'
    os.environ['XLA_DOWNCAST_BF16'] = '0'
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False

# --- Quick High-Res Dataset Downloader ---
class FastHDDataset(Dataset):
    def __init__(self, data_dir="hd_finetune_data"):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        self.images = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png'))]
        
        # Super-Crisp augmentations
        self.transform = transforms.Compose([
            transforms.RandomCrop((256, 256), pad_if_needed=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        return self.transform(img), 0

def download_div2k(data_dir="hd_finetune_data"):
    """Downloads a subset of the DIV2K High-Res dataset automatically."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    images = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png'))]
    if len(images) > 0:
        print(f"[*] Found {len(images)} HD images. Skipping download.")
        return

    print("[*] Downloading DIV2K High-Resolution Training Data (This contains extremely sharp edge info)...")
    url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
    zip_path = "DIV2K_train_HR.zip"
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url, zip_path)
    
    print("[*] Extracting HD Data...")
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if file.endswith('.png'):
                zip_ref.extract(file, ".")
                os.rename(file, os.path.join(data_dir, os.path.basename(file)))
    print("[+] HD Data acquired successfully.")

def run_reinforcement_finetune(args):
    device = xm.xla_device() if TPU_AVAILABLE else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Paradox Retina-Grade Fine-tuning Engine Activated on {device}", flush=True)

    download_div2k(args.data_dir)

    dataset = FastHDDataset(args.data_dir)
    # SPEED BOOST: Increased batch_size and enabled 8 workers for TPU Host saturation
    loader = DataLoader(
        dataset, batch_size=32, shuffle=True, 
        num_workers=8, drop_last=True, pin_memory=True
    )
    
    # 1. Load the Structurally Perfect Checkpoint
    model = LatentGenesisCore(latent_channels=64).to(device)
    if os.path.exists(args.checkpoint_path):
        print(f"[*] Loading foundation knowledge from {args.checkpoint_path}...")
        ckpt = torch.load(args.checkpoint_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        print("[!] No foundation checkpoint found! Train train_tpu.py first.")
        return

    # Keep a very low learning rate to preserve structure and only update textures
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    perc_engine = PerceptualLoss().to(device).eval()

    print("\n[*] Starting Reinforcement Phase. Learning High-Frequency Textures...")
    for epoch in range(args.epochs):
        model.train()
        
        train_device_loader = loader
        if TPU_AVAILABLE:
            train_device_loader = pl.ParallelLoader(loader, [device]).per_device_loader(device)
            
        pbar = tqdm(total=len(loader), desc=f"Finetune {epoch+1}/{args.epochs}", unit="batch")
        
        for images, _ in train_device_loader:
            if not TPU_AVAILABLE: images = images.to(device)
            optimizer.zero_grad()
            
            recon, mu, logvar = model(images)
            
            l1_l = F.l1_loss(recon, images)
            s_l  = torch.clamp(ssim_loss(recon, images), 0, 1)
            p_l  = torch.clamp(perc_engine(recon, images), 0, 100)
            
            # Lock the KLD tight to force the latent space to focus purely on texture accuracy
            kld_l = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            # THE PERFECT LEVEL FINETUNE:
            # Shift weight almost entirely to SSIM and Perceptual to burn in high-frequency details.
            loss = (l1_l * 1.0) + (s_l * 20.0) + (p_l * 2.0) + (kld_l * 0.0001)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            if TPU_AVAILABLE: xm.optimizer_step(optimizer)
            else: optimizer.step()
            
            pbar.update(1)

        pbar.close()

        # Overwrite the master with the Retina-grade version
        if TPU_AVAILABLE:
            xm.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, args.checkpoint_path)
        else:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, args.checkpoint_path)
            
        print(f"--- [MASTER SYNCHRONIZED] Retina Checkpoint Updated: {args.checkpoint_path} ---", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/universal_tpu_master.pth')
    parser.add_argument('--data_dir', type=str, default='hd_finetune_data')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-5) # 10x slower learning rate to protect structure
    args, _ = parser.parse_known_args()
    run_reinforcement_finetune(args)
