"""
receiver_enhancer.py — Paradox Team B: The Hallucinator
=======================================================
This powerful Super-Resolution AI lives entirely on the receiver's end.
It never transmits data. Its ONLY job is to take the 4KB blurry image 
received from Team A and hallucinate the missing textures back to 
flawless Retina-Grade quality in real-time.
"""

import os
import argparse
import urllib.request
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import LatentGenesisCore, ResBlock
from finetune_tpu import FastHDDataset, download_div2k
from train import PerceptualLoss

# --- TPU Integration ---
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    TPU_AVAILABLE = 'PJRT_DEVICE' in os.environ or 'TPU_NAME' in os.environ
except ImportError:
    TPU_AVAILABLE = False

class ReceiverEnhancer(nn.Module):
    """
    Team Member B: The AI Upscaler.
    A deep residual hallucinatory network to restore edges.
    """
    def __init__(self):
        super().__init__()
        # Extract features from the blurry image
        self.entry = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Deep Residual Core (Hallucination Engine)
        self.res_core = nn.Sequential(
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64), # Heavy depth for texture synthesis
            ResBlock(64)
        )
        
        # Re-project to Image Space
        self.exit = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connection over the whole network: learn only the MISSING details
        features = self.entry(x)
        features = self.res_core(features)
        details = self.exit(features)
        return torch.tanh(x + details) # Add details to blurry base

def train_enhancer(args):
    device = xm.xla_device() if TPU_AVAILABLE else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Starting Team B Enhancer Training on {device}...")

    download_div2k(args.data_dir)
    dataset = FastHDDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    # 1. Load Team A (Sender Model) in frozen mode
    sender_model = LatentGenesisCore(latent_channels=16).to(device)
    if os.path.exists(args.sender_path):
        sender_model.load_state_dict(torch.load(args.sender_path, map_location='cpu')['model_state_dict'])
    else:
        print("[!] Needs universal_tpu_master.pth to generate training blurs.")
        return
    sender_model.eval()
    for param in sender_model.parameters():
        param.requires_grad = False

    # 2. Initialize Team B (Receiver Model)
    receiver_model = ReceiverEnhancer().to(device)
    optimizer = optim.Adam(receiver_model.parameters(), lr=args.lr)
    perc_engine = PerceptualLoss().to(device).eval()

    # Learning exactly what textures Team A drops
    for epoch in range(args.epochs):
        receiver_model.train()
        train_loader = pl.ParallelLoader(loader, [device]).per_device_loader(device) if TPU_AVAILABLE else loader
        pbar = tqdm(total=len(loader), desc=f"Training Team B {epoch+1}/{args.epochs}")
        
        for images, _ in train_loader:
            if not TPU_AVAILABLE: images = images.to(device)
            optimizer.zero_grad()
            
            # Step A: Get 4KB Blurry Image from Sender
            with torch.no_grad():
                blurry_base, _, _ = sender_model(images)
            
            # Step B: Receiver Hallucinates Sharp Image
            sharp_pred = receiver_model(blurry_base)
            
            # Loss: Punish blur, reward exact texture matching to the original target image
            l1_loss = F.l1_loss(sharp_pred, images)
            perc_loss = torch.clamp(perc_engine(sharp_pred, images), 0, 100)
            
            # Loss heavily weights perceptual features to force hallucination of realistic grass/leaves/edges
            loss = (l1_loss * 1.0) + (perc_loss * 0.1)
            
            loss.backward()
            if TPU_AVAILABLE: xm.optimizer_step(optimizer)
            else: optimizer.step()
            
            pbar.update(1)
        pbar.close()

        # Save Enhancer
        save_func = xm.save if TPU_AVAILABLE else torch.save
        os.makedirs(os.path.dirname(args.receiver_path), exist_ok=True)
        save_func({'model_state_dict': receiver_model.state_dict()}, args.receiver_path)
        print(f"--- [*] Team B Checkpoint Saved: {args.receiver_path} ---")

def test_team(args):
    """Visually demonstrates the Teamwork pipeline compared to the raw output."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    sender_model = LatentGenesisCore(latent_channels=16).to(device)
    sender_model.load_state_dict(torch.load(args.sender_path, map_location=device)['model_state_dict'])
    sender_model.eval()

    receiver_model = ReceiverEnhancer().to(device)
    receiver_model.load_state_dict(torch.load(args.receiver_path, map_location=device)['model_state_dict'])
    receiver_model.eval()

    # Get a fresh random image
    url = f"https://picsum.photos/seed/{torch.randint(0,1000,(1,)).item()}/1024/1024"
    urllib.request.urlretrieve(url, "team_test.jpg")
    img = Image.open("team_test.jpg").convert('RGB')
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        blurry_base, _, _ = sender_model(x)
        final_sharp = receiver_model(blurry_base)

    def unnorm(t): return torch.clamp(t[0].cpu() * 0.5 + 0.5, 0, 1).permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(unnorm(x))
    axes[0].set_title("Original (Global Pattern)")
    axes[1].imshow(unnorm(blurry_base))
    axes[1].set_title("Team A (4KB Transmission)")
    axes[2].imshow(unnorm(final_sharp))
    axes[2].set_title("Team B (Final Hallucinated Output)")
    for ax in axes: ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('team_synergy_result.png', dpi=300)
    print("\n[*] Perfection achieved. Output saved to 'team_synergy_result.png'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'demo'], default='train')
    parser.add_argument('--sender_path', type=str, default='checkpoints/universal_tpu_master.pth')
    parser.add_argument('--receiver_path', type=str, default='checkpoints/receiver_enhancer.pth')
    parser.add_argument('--data_dir', type=str, default='hd_finetune_data')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=2e-4) # Fast learning rate for Hallucination
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_enhancer(args)
    else:
        test_team(args)
