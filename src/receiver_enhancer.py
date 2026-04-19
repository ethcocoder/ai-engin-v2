"""
receiver_enhancer.py — Paradox ESRGAN (Team B Elite Hallucinator)
================================================================
State-of-the-Art Adversarial Super Resolution Engine.
Uses Residual-in-Residual Dense Blocks (RRDB) and a PatchGAN Discriminator
to enforce photo-realistic hallucination of missing 4KB high frequencies.
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

from model import LatentGenesisCore
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


# =====================================================================
# 1. ELITE GENERATOR: RRDBNet (Residual in Residual Dense Block)
# =====================================================================
class DenseBlock(nn.Module):
    def __init__(self, channels=64, growth=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels + growth, growth, 3, 1, 1)
        self.conv3 = nn.Conv2d(channels + 2*growth, growth, 3, 1, 1)
        self.conv4 = nn.Conv2d(channels + 3*growth, growth, 3, 1, 1)
        self.conv5 = nn.Conv2d(channels + 4*growth, channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x # Residual scaling

class RRDB(nn.Module):
    """Residual in Residual Dense Block"""
    def __init__(self, channels=64, growth=32):
        super().__init__()
        self.RDB1 = DenseBlock(channels, growth)
        self.RDB2 = DenseBlock(channels, growth)
        self.RDB3 = DenseBlock(channels, growth)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x # Residual scaling

class EliteHallucinator(nn.Module):
    """The Ultimate 25MB+ Team B Generator"""
    def __init__(self, in_channels=3, out_channels=3, nf=64, nb=12): # 12 RRDB Blocks!
        super().__init__()
        self.conv_first = nn.Conv2d(in_channels, nf, 3, 1, 1)
        self.RRDB_trunk = nn.Sequential(*[RRDB(nf, 32) for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, out_channels, 3, 1, 1)
        )

    def forward(self, x_blurry):
        feat = self.conv_first(x_blurry)
        trunk = self.trunk_conv(self.RRDB_trunk(feat))
        feat = feat + trunk
        residual_details = self.conv_last(feat)
        return torch.tanh(x_blurry + residual_details) 


# =====================================================================
# 2. ELITE DISCRIMINATOR: VGG-Style Adversarial Network
# =====================================================================
class VGGDiscriminator(nn.Module):
    """Screams 'FAKE' if Team B generates blurry or mathematically average textures"""
    def __init__(self, in_channels=3):
        super().__init__()
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 3, stride=2, padding=1)]
            if normalize: layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Conv2d(512, 1, 3, 1, 1) # Single output map rating reality vs fake
        )

    def forward(self, img):
        return self.model(img)

# =====================================================================
# 3. TRAINING ENGINE
# =====================================================================
def train_gan_enhancer(args):
    device = xm.xla_device() if TPU_AVAILABLE else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Igniting Elite GAN Training Protocol on {device}...")

    download_div2k(args.data_dir)
    dataset = FastHDDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    sender_model = LatentGenesisCore(latent_channels=16).to(device)
    if os.path.exists(args.sender_path):
        sender_model.load_state_dict(torch.load(args.sender_path, map_location='cpu')['model_state_dict'])
    sender_model.eval()
    for param in sender_model.parameters(): param.requires_grad = False

    netG = EliteHallucinator().to(device)
    netD = VGGDiscriminator().to(device)
    
    optG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.9, 0.999))
    optD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.9, 0.999))
    
    perc_engine = PerceptualLoss().to(device).eval()
    criterion_gan = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        netG.train(); netD.train()
        train_loader = pl.ParallelLoader(loader, [device]).per_device_loader(device) if TPU_AVAILABLE else loader
        pbar = tqdm(total=len(loader), desc=f"Elite GAN {epoch+1}/{args.epochs}")
        
        for real_imgs, _ in train_loader:
            if not TPU_AVAILABLE: real_imgs = real_imgs.to(device)
            
            with torch.no_grad():
                blurry_base, _, _ = sender_model(real_imgs)
            
            # --- 1. Train Discriminator ---
            optD.zero_grad()
            fake_imgs = netG(blurry_base)
            
            pred_real = netD(real_imgs)
            pred_fake = netD(fake_imgs.detach())
            
            loss_D_real = criterion_gan(pred_real, torch.ones_like(pred_real))
            loss_D_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))
            loss_D = (loss_D_real + loss_D_fake) / 2
            
            loss_D.backward()
            if TPU_AVAILABLE: xm.optimizer_step(optD)
            else: optD.step()

            # --- 2. Train Generator ---
            optG.zero_grad()
            pred_fake_for_G = netD(fake_imgs) # re-eval since D updated
            
            # Adversarial Loss (trick the discriminator!)
            loss_G_adv = criterion_gan(pred_fake_for_G, torch.ones_like(pred_fake_for_G))
            # Perceptual Loss (match the high level visual features)
            loss_G_perc = torch.clamp(perc_engine(fake_imgs, real_imgs), 0, 100)
            # L1 Loss (maintain strict color/structural math)
            loss_G_l1 = F.l1_loss(fake_imgs, real_imgs)
            
            # The Magic Formula for Elite ESRGAN
            loss_G = (loss_G_l1 * 1.0) + (loss_G_perc * 0.1) + (loss_G_adv * 0.005)
            
            loss_G.backward()
            if TPU_AVAILABLE: xm.optimizer_step(optG)
            else: optG.step()
            
            pbar.set_postfix(D=f"{loss_D.item():.3f}", G=f"{loss_G.item():.3f}")
            pbar.update(1)
        pbar.close()

        save_func = xm.save if TPU_AVAILABLE else torch.save
        os.makedirs(os.path.dirname(args.receiver_path), exist_ok=True)
        save_func({'model_state_dict': netG.state_dict()}, args.receiver_path)

def test_elite(args):
    """Visually demonstrates the Adversarial Hallucination."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    sender_model = LatentGenesisCore(latent_channels=16).to(device)
    sender_model.load_state_dict(torch.load(args.sender_path, map_location=device)['model_state_dict'])
    sender_model.eval()

    receiver_model = EliteHallucinator().to(device)
    receiver_model.load_state_dict(torch.load(args.receiver_path, map_location=device)['model_state_dict'])
    receiver_model.eval()

    url = f"https://picsum.photos/seed/{torch.randint(0,1000,(1,)).item()}/1024/1024"
    urllib.request.urlretrieve(url, "elite_test.jpg")
    img = Image.open("elite_test.jpg").convert('RGB')
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
    axes[1].set_title("Team A (4KB Transmission - Blurry)")
    axes[2].imshow(unnorm(final_sharp))
    axes[2].set_title("Team B (Elite GAN Hallucinatory Restoration)")
    for ax in axes: ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('elite_gan_synergy.png', dpi=300)
    print("\n[*] Elite GAN Perfection achieved. Output saved to 'elite_gan_synergy.png'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'demo'], default='train')
    parser.add_argument('--sender_path', type=str, default='checkpoints/universal_tpu_master.pth')
    parser.add_argument('--receiver_path', type=str, default='checkpoints/elite_enhancer.pth')
    parser.add_argument('--data_dir', type=str, default='hd_finetune_data')
    parser.add_argument('--batch_size', type=int, default=8) # Lowered because RRDB takes memory!
    parser.add_argument('--epochs', type=int, default=100) # Full GAN training takes time
    parser.add_argument('--lr', type=float, default=1e-4) 
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_gan_enhancer(args)
    else:
        test_elite(args)
