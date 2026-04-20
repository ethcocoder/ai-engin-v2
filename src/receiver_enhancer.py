"""
receiver_enhancer.py — Paradox ESRGAN (Team B Elite Hallucinator)
================================================================
State-of-the-Art Adversarial Super Resolution Engine.
Rebuilt from the ground up for TPU/BF16 stability.
"""

import os
# --- CRITICAL FIX: Kill lingering BF16 environment variables from the previous script ---
os.environ['XLA_USE_BF16'] = '0'
os.environ['XLA_DOWNCAST_BF16'] = '0'

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
import torchvision.models as models

# --- Safe Perceptual Loss for BF16 ---
class SafePerceptualLoss(nn.Module):
    """Uses L1 instead of MSE to mathematically prevent BF16 squaring overflow"""
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*vgg[:4])   
        self.slice2 = nn.Sequential(*vgg[4:9])  
        self.slice3 = nn.Sequential(*vgg[9:16]) 
        for param in self.parameters():
            param.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        x = (x * 0.5 + 0.5 - self.mean) / self.std
        y = (y * 0.5 + 0.5 - self.mean) / self.std
        
        x_f1, y_f1 = self.slice1(x), self.slice1(y)
        x_f2, y_f2 = self.slice2(x_f1), self.slice2(y_f1)
        x_f3, y_f3 = self.slice3(x_f2), self.slice3(y_f2)
        
        # CRITICAL: L1 Loss does not square. 
        # MSE squares numbers. In BF16, a difference of 256 squared is 65536 -> NaN!
        return F.l1_loss(x_f1, y_f1) + F.l1_loss(x_f2, y_f2) + F.l1_loss(x_f3, y_f3)


# --- TPU Integration ---
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    TPU_AVAILABLE = 'PJRT_DEVICE' in os.environ or 'TPU_NAME' in os.environ
except ImportError:
    TPU_AVAILABLE = False


# =====================================================================
# 1. ELITE GENERATOR (BF16 & TPU Stabilized)
# =====================================================================
class StableDenseBlock(nn.Module):
    """Dense Block with InstanceNorm -> Required for BF16 stability without pretraining"""
    def __init__(self, channels=64, growth=32):
        super().__init__()
        def conv_layer(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
                nn.InstanceNorm2d(out_c),
                nn.LeakyReLU(0.2, True)
            )
        self.conv1 = conv_layer(channels, growth)
        self.conv2 = conv_layer(channels + growth, growth)
        self.conv3 = conv_layer(channels + 2*growth, growth)
        self.conv4 = conv_layer(channels + 3*growth, growth)
        self.conv5 = nn.Conv2d(channels + 4*growth, channels, 3, 1, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x # Residual scaling

class StableRRDB(nn.Module):
    def __init__(self, channels=64, growth=32):
        super().__init__()
        self.RDB1 = StableDenseBlock(channels, growth)
        self.RDB2 = StableDenseBlock(channels, growth)
        self.RDB3 = StableDenseBlock(channels, growth)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x 

class EliteHallucinator(nn.Module):
    """The Ultimate Team B Generator"""
    def __init__(self, in_channels=3, out_channels=3, nf=64, nb=6):
        super().__init__()
        self.conv_first = nn.Conv2d(in_channels, nf, 3, 1, 1)
        self.RRDB_trunk = nn.Sequential(*[StableRRDB(nf, 32) for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, out_channels, 3, 1, 1)
        )
        
        # Zero-Init to start safe
        nn.init.constant_(self.conv_last[-1].weight, 0)
        nn.init.constant_(self.conv_last[-1].bias, 0)

    def forward(self, x_blurry):
        feat = self.conv_first(x_blurry)
        trunk = self.trunk_conv(self.RRDB_trunk(feat))
        feat = feat + trunk
        residual_details = self.conv_last(feat)
        return torch.tanh(x_blurry + residual_details) 


# =====================================================================
# 2. ELITE DISCRIMINATOR: Stable PatchGAN
# =====================================================================
class StablePatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.utils.spectral_norm(nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1, bias=not normalize))]
            if normalize: layers.append(nn.InstanceNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.utils.spectral_norm(nn.Conv2d(512, 1, 3, 1, 1)) 
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
    
    sender_model = LatentGenesisCore(latent_channels=64).to(device)
    if os.path.exists(args.sender_path):
        sender_model.load_state_dict(torch.load(args.sender_path, map_location='cpu')['model_state_dict'])
    sender_model.eval()
    for param in sender_model.parameters(): param.requires_grad = False

    netG = EliteHallucinator(nb=args.nb).to(device)
    netD = StablePatchDiscriminator().to(device)
    
    # CRITICAL BF16 FIX: eps=1e-4 prevents Adam division by zero underflow on TPUs!
    optG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-4)
    optD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-4)
    
    perc_engine = SafePerceptualLoss().to(device).eval()
    criterion_gan = nn.BCEWithLogitsLoss() # LogSumExp prevents BF16 squaring overflow

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
            nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.0)
            
            if TPU_AVAILABLE: xm.optimizer_step(optD)
            else: optD.step()

            # --- 2. Train Generator ---
            optG.zero_grad()
            
            # Regenerate images to keep XLA graph clean
            fake_imgs_for_G = netG(blurry_base) 
            pred_fake_for_G = netD(fake_imgs_for_G) 
            
            # Losses
            loss_G_adv = criterion_gan(pred_fake_for_G, torch.ones_like(pred_fake_for_G))
            loss_G_perc = perc_engine(fake_imgs_for_G, real_imgs)
            loss_G_l1 = F.l1_loss(fake_imgs_for_G, real_imgs)
            
            # G needs a heavy L1 anchor initially to avoid exploding
            loss_G = (loss_G_l1 * 10.0) + (loss_G_perc * 0.1) + (loss_G_adv * 0.1)
            
            loss_G.backward()
            nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)
            
            if TPU_AVAILABLE: xm.optimizer_step(optG)
            else: optG.step()
            
            if TPU_AVAILABLE: torch_xla.sync() 
            
            pbar.set_postfix(D=f"{loss_D.item():.3f}", G=f"{loss_G.item():.3f}")
            pbar.update(1)
        pbar.close()

        save_func = xm.save if TPU_AVAILABLE else torch.save
        os.makedirs(os.path.dirname(args.receiver_path), exist_ok=True)
        save_func({'model_state_dict': netG.state_dict()}, args.receiver_path)

def test_elite(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    sender_model = LatentGenesisCore(latent_channels=64).to(device)
    sender_model.load_state_dict(torch.load(args.sender_path, map_location=device)['model_state_dict'])
    sender_model.eval()

    receiver_model = EliteHallucinator(nb=args.nb).to(device)
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
    axes[0].set_title("Original")
    axes[1].imshow(unnorm(blurry_base))
    axes[1].set_title("Team A (Blurry)")
    axes[2].imshow(unnorm(final_sharp))
    axes[2].set_title("Team B (Elite Sharp)")
    for ax in axes: ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('elite_gan_synergy.png', dpi=300)
    print("\n[*] Elite Analysis Complete. Output saved to 'elite_gan_synergy.png'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'demo'], default='train')
    parser.add_argument('--sender_path', type=str, default='checkpoints/universal_tpu_master.pth')
    parser.add_argument('--receiver_path', type=str, default='checkpoints/elite_enhancer.pth')
    parser.add_argument('--data_dir', type=str, default='hd_finetune_data')
    parser.add_argument('--batch_size', type=int, default=2) 
    parser.add_argument('--nb', type=int, default=6) # Safe memory limit
    parser.add_argument('--epochs', type=int, default=20) 
    parser.add_argument('--lr', type=float, default=1e-4) # Higher LR supported now
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_gan_enhancer(args)
    else:
        test_elite(args)
