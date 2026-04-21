"""
receiver_enhancer.py — Paradox ESRGAN (Team B - Sovereign Bespoke Edition)
==========================================================================
Engineered explicitly for hyper-rapid 20-Epoch 16KB texture hallucination.
Deploys Gram-Matrix Style Fusion and High-Frequency Edge Extraction to 
shatter the L1 blur-trap and evade VGG worm artifacts permanently.
"""

import os
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

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    TPU_AVAILABLE = 'PJRT_DEVICE' in os.environ or 'TPU_NAME' in os.environ
except ImportError:
    TPU_AVAILABLE = False


class EliteStyleFusionEngine(nn.Module):
    """
    Overcomes the VGG 'worm cheat' by capturing the raw texture *distributions* (Gram Matrix)
    instead of just point-to-point edge maps. Forces rapid generation of real 
    HD textures (wood, leaves, sky) regardless of spatial shifting.
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*vgg[:4])   # ReLU1_2
        self.slice2 = nn.Sequential(*vgg[4:9])  # ReLU2_2
        self.slice3 = nn.Sequential(*vgg[9:16]) # ReLU3_3
        self.slice4 = nn.Sequential(*vgg[16:23])# ReLU4_3
        for param in self.parameters():
            param.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def gram_matrix(self, input):
        b, c, h, w = input.size()
        features = input.view(b, c, h * w)
        G = torch.bmm(features, features.transpose(1, 2))
        return G.div(c * h * w)

    def forward(self, pred, target):
        pr = (pred * 0.5 + 0.5 - self.mean) / self.std
        tr = (target * 0.5 + 0.5 - self.mean) / self.std
        
        pr_f1 = self.slice1(pr); tr_f1 = self.slice1(tr)
        pr_f2 = self.slice2(pr_f1); tr_f2 = self.slice2(tr_f1)
        pr_f3 = self.slice3(pr_f2); tr_f3 = self.slice3(tr_f2)
        pr_f4 = self.slice4(pr_f3); tr_f4 = self.slice4(tr_f3)
        
        # Content Loss: Keeps the deep semantics (objects) locked.
        loss_content = F.l1_loss(pr_f4, tr_f4)
        
        # Style Loss: Explodes the speed at which real micro-textures synthesize
        loss_style = F.l1_loss(self.gram_matrix(pr_f1), self.gram_matrix(tr_f1)) + \
                     F.l1_loss(self.gram_matrix(pr_f2), self.gram_matrix(tr_f2)) + \
                     F.l1_loss(self.gram_matrix(pr_f3), self.gram_matrix(tr_f3))
                     
        return loss_content, loss_style


class HighFrequencyEdgeLoss(nn.Module):
    """
    Penalizes only the edges. Forces the AI to generate high-frequency sharpness 
    without falling into the 'blurry average' L1 trap.
    """
    def __init__(self):
        super().__init__()
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.register_buffer('kernel', laplacian.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1))

    def forward(self, pred, target):
        pred_edge = F.conv2d(pred, self.kernel, padding=1, groups=3)
        target_edge = F.conv2d(target, self.kernel, padding=1, groups=3)
        return F.l1_loss(pred_edge, target_edge)


# =====================================================================
# 1. ELITE GENERATOR: Unchained Speed
# =====================================================================
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class EliteHallucinator(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=6):
        super().__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = nn.Sequential(*[RRDB(nf=nf, gc=32) for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        )

    def forward(self, x):
        feat = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(feat))
        feat = feat + trunk
        residual = self.conv_last(feat)
        # CRITICAL UNCHAIN: Removed the 10% (.1) bottleneck multiplier. 
        # For a 20-epoch speedrun, the AI must be permitted to blast full power immediately.
        return torch.clamp(x + residual, -1.0, 1.0)


# =====================================================================
# 2. ELITE DISCRIMINATOR
# =====================================================================
class StableDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        def block(in_feat, out_feat, stride=1, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 3, stride=stride, padding=1, bias=not normalize)]
            if normalize: 
                layers.append(nn.GroupNorm(8, out_feat, eps=1e-4))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, 64, stride=2, normalize=False),
            *block(64, 128, stride=2),
            *block(128, 256, stride=2),
            *block(256, 512, stride=2),
            nn.Conv2d(512, 1, 3, 1, 1) 
        )

    def forward(self, img):
        return self.model(img)


# =====================================================================
# 3. TRAINING ENGINE
# =====================================================================
def train_gan_enhancer(args):
    device = torch_xla.device() if TPU_AVAILABLE else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Igniting Bespoke Elite GAN on {device}...")

    download_div2k(args.data_dir)
    dataset = FastHDDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    sender_model = LatentGenesisCore(latent_channels=64).to(device)
    if os.path.exists(args.sender_path):
        sender_model.load_state_dict(torch.load(args.sender_path, map_location='cpu')['model_state_dict'])
    sender_model.eval()
    for param in sender_model.parameters(): param.requires_grad = False

    netG = EliteHallucinator(nb=args.nb).to(device)
    netD = StableDiscriminator().to(device)
    
    # UNCHAINED LR: Generator boosted to learn 2x faster in the 20 epoch sprint.
    optG = optim.Adam(netG.parameters(), lr=args.lr * 2.0, betas=(0.5, 0.999), eps=1e-4)
    optD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999), eps=1e-4)
    
    style_engine = EliteStyleFusionEngine().to(device).eval()
    hf_engine = HighFrequencyEdgeLoss().to(device).eval()
    criterion_gan = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        netG.train(); netD.train()
        train_loader = pl.ParallelLoader(loader, [device]).per_device_loader(device) if TPU_AVAILABLE else loader
        pbar = tqdm(total=len(loader), desc=f"Elite GAN {epoch+1}/{args.epochs}")
        
        for real_imgs, _ in train_loader:
            if not TPU_AVAILABLE: real_imgs = real_imgs.to(device)
            
            with torch.no_grad():
                blurry_base, _, _ = sender_model(real_imgs)
            
            # --- Train Discriminator ---
            optD.zero_grad()
            fake_imgs = netG(blurry_base)
            
            pred_real = netD(real_imgs)
            pred_fake = netD(fake_imgs.detach())
            
            target_real = torch.full_like(pred_real, 0.9)
            target_fake = torch.zeros_like(pred_fake)
            
            loss_D_real = criterion_gan(pred_real, target_real)
            loss_D_fake = criterion_gan(pred_fake, target_fake)
            loss_D = (loss_D_real + loss_D_fake) / 2
            
            loss_D.backward()
            nn.utils.clip_grad_norm_(netD.parameters(), max_norm=0.5)
            
            if TPU_AVAILABLE: xm.optimizer_step(optD)
            else: optD.step()

            # --- Train Generator ---
            optG.zero_grad()
            
            fake_imgs_for_G = netG(blurry_base) 
            pred_fake_for_G = netD(fake_imgs_for_G) 
            
            target_real_G = torch.ones_like(pred_fake_for_G)
            
            loss_G_adv = criterion_gan(pred_fake_for_G, target_real_G)
            loss_content, loss_style = style_engine(fake_imgs_for_G, real_imgs)
            loss_edge = hf_engine(fake_imgs_for_G, real_imgs)
            loss_l1 = F.l1_loss(fake_imgs_for_G, real_imgs)
            
            # THE BESPOKE FORMULA:
            # - Remove massive L1 anchor (It causes mathematically mandated average blur).
            # - Style * 10.0 synthesizes high-res HD textures directly from patterns.
            # - Edge * 5.0 forces physical sharpness out of the Blur hole.
            # - Adv * 0.05 drives local reality without causing neon hacks.
            loss_G = (loss_l1 * 1.0) + (loss_content * 1.0) + (loss_style * 10.0) + (loss_edge * 5.0) + (loss_G_adv * 0.05)
            
            loss_G.backward()
            nn.utils.clip_grad_norm_(netG.parameters(), max_norm=0.5)
            
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
    plt.savefig('elite_gan_synergy.png', formatted_dpi=300)
    print("\n[*] Elite Analysis Complete. Output saved to 'elite_gan_synergy.png'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'demo'], default='train')
    parser.add_argument('--sender_path', type=str, default='checkpoints/universal_tpu_master.pth')
    parser.add_argument('--receiver_path', type=str, default='checkpoints/elite_enhancer.pth')
    parser.add_argument('--data_dir', type=str, default='hd_finetune_data')
    parser.add_argument('--batch_size', type=int, default=2) 
    parser.add_argument('--nb', type=int, default=6) 
    parser.add_argument('--epochs', type=int, default=20) 
    parser.add_argument('--lr', type=float, default=1e-4) 
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_gan_enhancer(args)
    else:
        test_elite(args)
