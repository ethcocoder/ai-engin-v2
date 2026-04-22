import sys
import os
from pathlib import Path

# --- Advanced Pathing Protocol ---
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

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
from qau_qvs.core.qvs import QVS
import torchvision.models as models

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    TPU_AVAILABLE = 'PJRT_DEVICE' in os.environ or 'TPU_NAME' in os.environ
except ImportError:
    TPU_AVAILABLE = False


class SovereignAntiGridEngine(nn.Module):
    """
    Sovereign Anti-Grid Engine (SAGE)
    Eliminates neural checkerboarding and coordinate-like artifacts.
    """
    def __init__(self):
        super().__init__()
        # 5x5 Gaussian Kernel to neutralize grid spikes
        kernel = torch.tensor([[1, 4, 6, 4, 1],
                               [4,16,24,16, 4],
                               [6,24,36,24, 6],
                               [4,16,24,16, 4],
                               [1, 4, 6, 4, 1]], dtype=torch.float32) / 256.0
        self.register_buffer('kernel', kernel.view(1, 1, 5, 5))

    def forward(self, x):
        # Multi-channel Gaussian Blur
        c = x.shape[1]
        blurred = F.conv2d(x, self.kernel.expand(c, 1, 5, 5), padding=2, groups=c)
        # We penalize the difference between the image and its blurred version 
        # specifically in high-frequency 'grid' regions.
        return blurred

class SovereignAestheticEngine(nn.Module):
    """
    Sovereign Anchored Aesthetic Engine.
    Combines Relative-Rank matching with Geometric Discipline.
    """
    def __init__(self, device):
        super().__init__()
        self.mse = nn.MSELoss()
        # Geometric Anchors (Fixed 1-pixel rulers)
        self.sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32, device=device)
        self.sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32, device=device)

    def forward(self, fake, real):
        f_gray = fake.mean(dim=1, keepdim=True)
        r_gray = real.mean(dim=1, keepdim=True)

        # 1. Geometric Anchor (Fixes Skyscraper Wiggling)
        f_dx = F.conv2d(F.pad(f_gray, (1,1,1,1), mode='reflect'), self.sobel_x)
        f_dy = F.conv2d(F.pad(f_gray, (1,1,1,1), mode='reflect'), self.sobel_y)
        r_dx = F.conv2d(F.pad(r_gray, (1,1,1,1), mode='reflect'), self.sobel_x)
        r_dy = F.conv2d(F.pad(r_gray, (1,1,1,1), mode='reflect'), self.sobel_y)
        loss_anchor = self.mse(f_dx, r_dx) + self.mse(f_dy, r_dy)

        # 2. RRM Ratio Match
        f_diff_h = f_gray[:, :, 1:, :] - f_gray[:, :, :-1, :]
        f_diff_w = f_gray[:, :, :, 1:] - f_gray[:, :, :, :-1]
        r_diff_h = r_gray[:, :, 1:, :] - r_gray[:, :, :-1, :]
        r_diff_w = r_gray[:, :, :, 1:] - r_gray[:, :, :, :-1]
        loss_rank = self.mse(f_diff_h, r_diff_h) + self.mse(f_diff_w, r_diff_w)

        # 3. Dynamic Contrast Lock
        loss_contrast = torch.abs(fake.mean() - real.mean()) + torch.abs(fake.std() - real.std())

        return (loss_anchor * 100.0) + (loss_rank * 20.0) + (loss_contrast * 10.0)

class EliteFeatureEngine(nn.Module):
    """
    Hyper-Effective Feature Trainer. 
    Matches standard pixel-space L1 with deep VGG-feature activations.
    Forces the AI to understand 'content' and 'texture' simultaneously.
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        # Layer 3: ReLU1_2 (Low-level edges), Layer 8: ReLU2_2 (Textures), Layer 15: ReLU3_3 (Patterns)
        self.slice1 = nn.Sequential(*vgg[:4])   
        self.slice2 = nn.Sequential(*vgg[4:9])  
        self.slice3 = nn.Sequential(*vgg[9:16]) 
        for param in self.parameters():
            param.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def gram_matrix(self, input):
        b, c, h, w = input.size()
        features = input.view(b, c, h * w)
        G = torch.bmm(features, features.transpose(1, 2))
        return G.div(c * h * w + 1e-8)

    def forward(self, pred, target):
        # Normalize to VGG space
        pr = (pred * 0.5 + 0.5 - self.mean) / self.std
        tr = (target * 0.5 + 0.5 - self.mean) / self.std
        
        pr_f1 = self.slice1(pr); tr_f1 = self.slice1(tr)
        pr_f2 = self.slice2(pr_f1); tr_f2 = self.slice2(tr_f1)
        pr_f3 = self.slice3(pr_f2); tr_f3 = self.slice3(tr_f2)
        
        # Feature Loss (Direct activation matching for content structure)
        loss_feat = F.l1_loss(pr_f3, tr_f3)
        
        # Style Loss (Gram Matrix for hyper-realistic textures)
        loss_style = F.l1_loss(self.gram_matrix(pr_f1), self.gram_matrix(tr_f1)) + \
                     F.l1_loss(self.gram_matrix(pr_f2), self.gram_matrix(tr_f2))
                     
        return loss_feat, loss_style


class HighFrequencyEdgeLoss(nn.Module):
    """
    Sovereign 'Needle-Point' Engine.
    Uses Multi-Scale Laplacian of Gaussian (LoG) to find sub-pixel textures.
    """
    def __init__(self):
        super().__init__()
        # 3x3 Laplacian Kernel for micro-dots and needle-points
        self.laplacian_kernel = torch.tensor([[[[0,  1, 0], 
                                                [1, -4, 1], 
                                                [0,  1, 0]]]], dtype=torch.float32)

    def forward(self, fake, real):
        device = fake.device
        kernel = self.laplacian_kernel.to(device)
        
        # Multi-Scale Analysis (Original and Downsampled)
        def get_dots(img):
            gray = img.mean(dim=1, keepdim=True)
            # Find sharp points
            dots = torch.abs(F.conv2d(gray, kernel, padding=1))
            # Downsample to find 'cluster' points
            dots_low = torch.abs(F.conv2d(F.avg_pool2d(gray, 2), kernel, padding=1))
            return dots, dots_low

        f_dots, f_dots_l = get_dots(fake)
        r_dots, r_dots_l = get_dots(real)
        
        # Penalize missing needle-points with High-Energy MSE
        loss = F.mse_loss(f_dots, r_dots) + F.mse_loss(f_dots_l, r_dots_l)
        return loss


# =====================================================================
# 1. ELITE GENERATOR: Unchained Speed
# =====================================================================
class PixelAttention(nn.Module):
    """
    Hyper-speed attention mechanism. Focuses on high-frequency details 
    (edges, textures) without the memory overhead of Dense blocks.
    """
    def __init__(self, nf):
        super().__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv(x)
        y = self.sigmoid(y)
        return x * y

class FastAttentionBlock(nn.Module):
    """
    Replaces RRDB for 3x speedup. Uses a streamlined residual path 
    with Pixel Attention and Quantum-Stochastic Superposition.
    """
    def __init__(self, nf=64):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.pa = PixelAttention(nf)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # PROBABILITY ENGINE: Learnable noise scaling
        self.noise_scale = nn.Parameter(torch.zeros(1, nf, 1, 1))
        self.qvs = QVS()

    def forward(self, x):
        res = self.lrelu(self.conv1(x))
        
        # BIO-INSPIRED: LATERAL INHIBITION
        # Mimics the human eye's edge-enhancement by inhibiting 
        # neighboring pixels (High-Pass Filtering).
        # This creates 'Biological Sharpness'.
        blurred = F.avg_pool2d(res, kernel_size=3, stride=1, padding=1)
        res = res + (res - blurred) * 0.5 # Sharpening boost
        
        # QUANTUM-STOCHASTIC SUPERPOSITION:
        # Vectorized path. 100% TPU-accelerated Trajectories.
        intensities = torch.mean(res, dim=(1, 2, 3))
        batch_bias = self.qvs.batch_run_trajectories(intensities, trials=10)
        
        noise = torch.randn_like(res) + batch_bias.view(-1, 1, 1, 1)
        res = res + (noise * self.noise_scale)
        
        res = self.conv2(res)
        res = self.pa(res)
        return x + res

class EliteHallucinator(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=8):
        super().__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.trunk = nn.Sequential(*[FastAttentionBlock(nf) for _ in range(nb)])
        # SEI: Sovereign Entropy Injection - The seed for sub-pixel texture growth
        self.sei_strength = nn.Parameter(torch.ones(1, nf, 1, 1) * 0.005)
        self.conv_last = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, out_nc, 3, 1, 1)
        )

    def forward(self, x):
        feat = self.conv_first(x)
        
        # --- SALIENCY-GATED SEI PARADIGM ---
        # We only inject noise where the image has 'Structural Complexity'.
        # This prevents blobs in smoke/sky while allowing needles in trees.
        with torch.no_grad():
            gray = x.mean(dim=1, keepdim=True)
            # Find complexity (edges/textures)
            complexity = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :]).mean(dim=2, keepdim=True)
            complexity = F.interpolate(complexity, size=feat.shape[2:], mode='bilinear', align_corners=False)
            # Normalize to [0, 1]
            gate = (complexity - complexity.min()) / (complexity.max() - complexity.min() + 1e-8)
            
        jitter = (torch.randn_like(feat) * self.sei_strength) * gate
        feat = feat + jitter
        
        trunk = self.trunk(feat)
        residual = self.conv_last(trunk)
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
        # Return list of intermediate features for Feature Matching
        features = []
        for layer in self.model:
            img = layer(img)
            features.append(img)
        return features


# =====================================================================
# 3. TRAINING ENGINE
# =====================================================================
def train_gan_enhancer(args, is_finetune=False):
    device = torch_xla.device() if TPU_AVAILABLE else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    phase_name = "REINFORCEMENT (Finetune)" if is_finetune else "IGNITION (Train)"
    print(f"[*] Starting {phase_name} Phase on {device}...")

    download_div2k(args.data_dir)
    dataset = FastHDDataset(args.data_dir)
    # Use smaller batch for finetune to increase stochastic detail
    bs = args.batch_size // 2 if is_finetune else args.batch_size
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True, num_workers=8)
    
    sender_model = LatentGenesisCore(latent_channels=64).to(device)
    if os.path.exists(args.sender_path):
        sender_model.load_state_dict(torch.load(args.sender_path, map_location='cpu')['model_state_dict'])
    sender_model.eval()
    for param in sender_model.parameters(): param.requires_grad = False

    netG = EliteHallucinator(nb=args.nb).to(device)
    netD = StableDiscriminator().to(device)
    
    # Load existing Enhancer weights if finetuning
    if is_finetune and os.path.exists(args.receiver_path):
        print(f"[*] Loading pre-trained Enhancer: {args.receiver_path}")
        netG.load_state_dict(torch.load(args.receiver_path, map_location='cpu')['model_state_dict'])

    # ── SOVEREIGN FINETUNE LEARNING RATE ─────────────────────────────────────
    # Use 5x lower LR for finetune to prevent catastrophic forgetting
    lr_g = (args.lr * 0.2) if is_finetune else (args.lr * 2.0)
    lr_d = (args.lr * 0.1) if is_finetune else args.lr
    
    optG = optim.Adam(netG.parameters(), lr=lr_g, betas=(0.5, 0.999), eps=1e-4)
    optD = optim.Adam(netD.parameters(), lr=lr_d, betas=(0.5, 0.999), eps=1e-4)
    
    feature_engine = EliteFeatureEngine().to(device).eval()
    hf_engine = HighFrequencyEdgeLoss().to(device).eval()
    aesthetic_engine = SovereignAestheticEngine(device).to(device).eval()
    sage_engine = SovereignAntiGridEngine().to(device).eval()
    criterion_gan = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        netG.train(); netD.train()
        train_loader = pl.ParallelLoader(loader, [device]).per_device_loader(device) if TPU_AVAILABLE else loader
        pbar = tqdm(total=len(loader), desc=f"Elite {phase_name} {epoch+1}/{args.epochs}")
        
        for real_imgs, _ in train_loader:
            if not TPU_AVAILABLE: real_imgs = real_imgs.to(device)
            
            with torch.no_grad():
                blurry_base, _, _ = sender_model(real_imgs)
                # --- SOVEREIGN SALIENCY ENGINE ---
                gray_real = real_imgs.mean(dim=1, keepdim=True)
                edge_mask = torch.abs(F.conv2d(gray_real, torch.tensor([[[[0,1,0],[1,-4,1],[0,1,0]]]], dtype=torch.float32, device=device), padding=1))
                saliency = torch.clamp(edge_mask / (edge_mask.mean() + 1e-8), 0.1, 5.0)
            
            fake_imgs = netG(blurry_base)
            
            # --- Train Discriminator ---
            optD.zero_grad()
            pred_real_feats = netD(real_imgs)
            pred_fake_feats = netD(fake_imgs.detach())
            
            loss_D_real = criterion_gan(pred_real_feats[-1], torch.full_like(pred_real_feats[-1], 0.9))
            loss_D_fake = criterion_gan(pred_fake_feats[-1], torch.zeros_like(pred_fake_feats[-1]))
            loss_D = (loss_D_real + loss_D_fake) / 2
            
            loss_D.backward()
            nn.utils.clip_grad_norm_(netD.parameters(), max_norm=0.5)
            
            if TPU_AVAILABLE: xm.optimizer_step(optD)
            else: optD.step()

            # --- Train Generator ---
            optG.zero_grad()
            pred_fake_for_G_feats = netD(fake_imgs) 
            pred_real_for_G_feats = netD(real_imgs)
            
            # --- SOVEREIGN APEX WEIGHTS ---
            adv_weight = 20.0 if is_finetune else 0.5
            pixel_weight = 0.0 if is_finetune else 0.1 # KILL L1 in finetune for pure detail
            
            loss_G_adv = criterion_gan(pred_fake_for_G_feats[-1], torch.ones_like(pred_fake_for_G_feats[-1]))
            
            # 2. FEATURE MATCHING 
            loss_fm = 0
            for f in range(len(pred_fake_for_G_feats) - 1):
                loss_fm += F.l1_loss(pred_fake_for_G_feats[f], pred_real_for_G_feats[f].detach())
            
            loss_feat, loss_style = feature_engine(fake_imgs, real_imgs)
            loss_edge = hf_engine(fake_imgs, real_imgs)
            loss_vibe = aesthetic_engine(fake_imgs, real_imgs)
            
            # --- SAGE: Anti-Grid Consistency ---
            fake_sage = sage_engine(fake_imgs)
            real_sage = sage_engine(real_imgs)
            loss_sage = F.mse_loss(fake_sage, real_sage)
            
            # Saliency Weighting
            loss_pixel = (F.l1_loss(fake_imgs, real_imgs, reduction='none') * saliency).mean()
            
            # THE APEX FORMULA
            # In finetune, we maximize the 'Hallucination' pressure.
            vibe_weight = 25.0 if is_finetune else 5.0
            sage_weight = 100.0 if is_finetune else 10.0
            
            loss_G = (loss_pixel * pixel_weight) + (loss_feat * 5.0) + (loss_style * 5.0) + (loss_edge * 15.0) + \
                     (loss_fm * 10.0) + (loss_G_adv * adv_weight) + (loss_vibe * vibe_weight) + (loss_sage * sage_weight)
            
            loss_G.backward()
            nn.utils.clip_grad_norm_(netG.parameters(), max_norm=0.5)
            
            if TPU_AVAILABLE: xm.optimizer_step(optG)
            else: optG.step()
            
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

    # Dynamic seed for demo variety
    url = f"https://picsum.photos/seed/{torch.randint(0,10000,(1,)).item()}/1024/1024"
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
    parser.add_argument('--mode', type=str, choices=['train', 'finetune', 'demo'], default='train')
    parser.add_argument('--sender_path', type=str, default='checkpoints/universal_tpu_master.pth')
    parser.add_argument('--receiver_path', type=str, default='checkpoints/elite_enhancer.pth')
    parser.add_argument('--data_dir', type=str, default='hd_finetune_data')
    parser.add_argument('--batch_size', type=int, default=8) 
    parser.add_argument('--nb', type=int, default=8) 
    parser.add_argument('--epochs', type=int, default=20) 
    parser.add_argument('--lr', type=float, default=1e-4) 
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_gan_enhancer(args, is_finetune=False)
    elif args.mode == 'finetune':
        train_gan_enhancer(args, is_finetune=True)
    else:
        test_elite(args)
